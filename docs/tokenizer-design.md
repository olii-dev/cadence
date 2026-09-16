# Tokenizer design

Cadence uses an original, REMI-inspired event stream designed for polyphonic, multi-instrument MIDI:

- `TIME_SHIFT`: 24 positions per beat, split into bounded chunks for long silence
- `TRACK`: canonical MIDI voice identity, derived from source track plus channel
- `PROGRAM` or `DRUMS`
- `PITCH`, `VELOCITY`, `DURATION`
- global `TEMPO` and `TIME_SIGNATURE`

The fixed beat grid keeps rhythm explicit and makes attention over musical structure easier than raw MIDI delta-time bytes. Duration tokens avoid a fragile note-on/note-off pairing problem during generation. Track and program tokens preserve arrangements, including program changes and percussion.

Correctness target: canonical musical round-trip. After the intentional quantization and clipping rules, encoding a decoded sequence must return the exact token sequence. The current deterministic 100-file Lakh sample has 99 valid files, one corrupt file, and 99/99 exact canonical round trips. The corrupt-file rejection is expected because the official Lakh full archive documents that it contains several thousand likely-corrupt files.
