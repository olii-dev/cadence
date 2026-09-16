# Cadence

A small symbolic-music transformer built and trained from scratch. Cadence learns MIDI as a language: notes, rhythm, instrumentation, dynamics, tempo, and meter become tokens for a GPT-style next-token model.

**No pretrained weights. No fine-tuning. No copied reference implementation code.** Papers and specifications inform the design; all implementation here is original.

## Status

Early build: tokenizer, dataset pipeline, model, CPU smoke test, generation, and audio rendering are being implemented before any GPU training.

## Design

The tokenizer is REMI-inspired but polyphonic and multi-instrument aware. It uses a fixed 24-position-per-beat grid and loss-aware event tuples:

- time shift
- program or drums
- pitch
- quantized velocity
- duration
- tempo
- time signature

Token-to-MIDI decoding creates a canonical MIDI. The primary correctness invariant is `encode(decode(encode(song))) == encode(song)`, tested across tempo, meter, instruments, drums, overlapping notes, and long time shifts.

## Development

```bash
uv sync --extra dev
uv run pytest
```

## Data

The planned corpus is the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/), distributed by its curator under CC BY 4.0. Cadence does not redistribute the archive. Please cite Colin Raffel, *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching* (PhD thesis, 2016). The curator notes that he did not transcribe the MIDI files and that per-file authorship is often unavailable because MIDI copyright metadata is inconsistent. Training-data, model-release, and generated-sample review must therefore preserve that caveat rather than treating the collection-level license as perfect song-by-song provenance.

## License

Code license to be selected before the first public release. Dataset files, soundfonts, and generated artifacts retain their own terms.
