"""Streaming MIDI cleaning and tokenization pipeline."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import mido
import numpy as np
from tqdm import tqdm

from .tokenizer import MidiTokenizer


@dataclass
class DatasetStats:
    discovered: int = 0
    accepted: int = 0
    duplicates: int = 0
    broken: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    total_tokens: int = 0
    total_notes: int = 0


def canonical_digest(token_ids: list[int]) -> str:
    return sha256(np.asarray(token_ids, dtype=np.uint16).tobytes()).hexdigest()


def build_dataset(input_dir: Path, output_dir: Path, min_notes=16, max_tokens=131072, start=0, limit=None) -> DatasetStats:
    tokenizer = MidiTokenizer()
    all_paths = sorted(set(input_dir.rglob("*.mid")) | set(input_dir.rglob("*.midi")))
    paths = all_paths[start : start + limit if limit is not None else None]
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = {split: [] for split in ("train", "validation", "test")}
    manifests = {split: [] for split in shards}
    stats = DatasetStats(discovered=len(paths)); seen = set()
    token_lengths = []; note_counts = []; instruments = Counter(); meters = Counter(); tempos = Counter()
    for path in tqdm(paths, desc="Tokenizing MIDI"):
        try:
            midi = mido.MidiFile(path, clip=False)
            events = tokenizer.extract_events(midi)
            notes = [e for e in events if e.kind == "note"]
            if not events:
                stats.empty += 1; continue
            if len(notes) < min_notes:
                stats.too_short += 1; continue
            ids = tokenizer.encode_ids(midi)
            if len(ids) > max_tokens:
                stats.too_long += 1; continue
            digest = canonical_digest(ids)
            if digest in seen:
                stats.duplicates += 1; continue
            seen.add(digest)
        except (OSError, EOFError, ValueError, TypeError, KeyError, mido.KeySignatureError):
            stats.broken += 1; continue
        # Content-hash assignment is stable across machines, traversal order, and resumable shards.
        split_bucket = int(digest[:8], 16) % 100
        split = "train" if split_bucket < 98 else ("validation" if split_bucket == 98 else "test")
        offset = sum(len(x) + 1 for x in shards[split])
        shards[split].append(np.asarray(ids, dtype=np.uint16))
        manifests[split].append({"source": str(path.relative_to(input_dir)), "digest": digest, "offset": offset, "length": len(ids)})
        stats.accepted += 1; stats.total_tokens += len(ids); stats.total_notes += len(notes)
        token_lengths.append(len(ids)); note_counts.append(len(notes))
        for e in events:
            if e.kind == "note": instruments["drums" if e.values[4] else str(e.values[3])] += 1
            elif e.kind == "meter": meters[f"{e.values[0]}/{e.values[1]}"] += 1
            elif e.kind == "tempo": tempos[str(e.values[0])] += 1
    for split, songs in shards.items():
        flat = np.concatenate([np.append(s, tokenizer.token_to_id["EOS"]) for s in songs]) if songs else np.array([], dtype=np.uint16)
        np.save(output_dir / f"{split}.npy", flat)
        (output_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in manifests[split]))
    tokenizer.save_vocab(output_dir / "vocab.json")
    summary = asdict(stats) | {
        "source_total": len(all_paths), "range_start": start,
        "range_end": start + len(paths),
        "vocab_size": len(tokenizer.vocab),
        "token_length_percentiles": np.percentile(token_lengths, [5, 25, 50, 75, 95]).tolist() if token_lengths else [],
        "note_count_percentiles": np.percentile(note_counts, [5, 25, 50, 75, 95]).tolist() if note_counts else [],
        "top_programs": instruments.most_common(20), "top_meters": meters.most_common(10), "top_tempos": tempos.most_common(10),
    }
    (output_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--min-notes", type=int, default=16); parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--start", type=int, default=0, help="sorted source-file offset for resumable preprocessing")
    parser.add_argument("--limit", type=int, help="maximum source files in this shard")
    args = parser.parse_args(); print(build_dataset(args.input, args.output, args.min_notes, args.max_tokens, args.start, args.limit))

if __name__ == "__main__": main()
