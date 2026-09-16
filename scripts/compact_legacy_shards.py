"""Shrink completed legacy int64 NPY shards to raw uint16 in place."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation", "test")
CHUNK_ELEMENTS = 1_000_000


def expected_elements(shard: Path, split: str) -> int:
    return sum(json.loads(line)["length"] + 1 for line in (shard / f"{split}.jsonl").read_text().splitlines())


def compact_file(path: Path, expected: int) -> None:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"{path}: expected {expected} elements, found {array.shape}")
    if array.dtype != np.int64:
        raise ValueError(f"{path}: expected legacy int64, found {array.dtype}")
    if expected and (array.min() < 0 or array.max() > np.iinfo(np.uint16).max):
        raise ValueError(f"{path}: token id is outside uint16 range")
    source_offset = array.offset
    del array
    fd = os.open(path, os.O_RDWR)
    try:
        for start in range(0, expected, CHUNK_ELEMENTS):
            count = min(CHUNK_ELEMENTS, expected - start)
            raw = os.pread(fd, count * 8, source_offset + start * 8)
            if len(raw) != count * 8:
                raise OSError(f"{path}: short read at element {start}")
            packed = np.frombuffer(raw, dtype=np.int64).astype(np.uint16).tobytes()
            written = os.pwrite(fd, packed, start * 2)
            if written != len(packed):
                raise OSError(f"{path}: short write at element {start}")
        os.ftruncate(fd, expected * 2)
        os.fsync(fd)
    finally:
        os.close(fd)


def compact_shard(shard: Path) -> None:
    if not (shard / "stats.json").exists():
        raise ValueError(f"refusing unmarked shard: {shard}")
    complete = shard / "packed.json"
    if complete.exists():
        return
    in_progress = shard / ".packing"
    if in_progress.exists():
        raise RuntimeError(f"interrupted conversion; delete and recompute shard: {shard}")
    in_progress.write_text("legacy NPY conversion in progress\n")
    counts = {}
    try:
        for split in SPLITS:
            source = shard / f"{split}.npy"
            target = shard / f"{split}.bin"
            if not source.exists() or target.exists():
                raise ValueError(f"unexpected legacy files in {shard} for {split}")
            counts[split] = expected_elements(shard, split)
            compact_file(source, counts[split])
            source.rename(target)
        complete.write_text(json.dumps({"dtype": "uint16", "elements": counts}, indent=2))
        in_progress.unlink()
    except Exception:
        # The marker makes a partially overwritten shard impossible to mistake for valid.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for shard in sorted(args.root.glob("shard-*")):
        if (shard / "stats.json").exists():
            print(f"Compacting {shard}", flush=True)
            compact_shard(shard)


if __name__ == "__main__":
    main()
