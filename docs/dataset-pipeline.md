# Dataset pipeline

Cadence scans MIDI files in stable sorted order and supports bounded `--start`/`--limit` ranges so a full corpus build can resume shard by shard. Every accepted song is converted to `uint16` token IDs and assigned to train/validation/test from its canonical content hash, so splits do not change with machine, traversal order, or shard boundaries.

Current filters reject unreadable MIDI, empty files, songs under 16 note events, and sequences over 131,072 tokens. Canonical-token hashing catches musical duplicates that differ at the raw-file level.

## First 500 source files

- accepted: 484 (96.8%)
- corrupt/unreadable: 11
- too short: 3
- too long: 2
- notes: 1,824,911
- tokens: 9,699,984
- vocabulary: 1,759 tokens
- median song: 3,147 notes / 16,738 tokens
- 95th percentile: 9,958 notes / 51,738 tokens
- processing time on the initial CPU worker: 75 seconds

These are an early deterministic slice, not final corpus-wide estimates. The full source tree contains 178,561 `.mid` files after extraction, while the official dataset page describes 176,581 unique files. The difference needs investigation before final reporting and may include archive/path variants or extraction-count differences.
