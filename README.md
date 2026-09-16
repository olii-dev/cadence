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

The planned corpus is the Lakh MIDI Dataset. The dataset is downloaded separately and is not redistributed by this repository. MIDI copyright status can vary by file; model releases and generated samples require a separate legal/data review.

## License

Code license to be selected before the first public release. Dataset files, soundfonts, and generated artifacts retain their own terms.
