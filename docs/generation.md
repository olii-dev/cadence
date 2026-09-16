# Generation

Cadence samples autoregressively with temperature and top-k controls, then decodes the token sequence into canonical MIDI. Generation applies a small state-machine grammar before sampling:

1. event boundary: time/tempo/meter, a new track-note tuple, or end
2. track
3. program or drums
4. pitch
5. velocity
6. duration

Masking impossible next-token classes prevents incomplete note fragments and keeps output parseable without changing the model weights. It is structural validity, not a claim that the notes are musically good.

Audio rendering is optional and external: point `cadence-generate --soundfont` at a compatible SoundFont when FluidSynth is installed. Cadence does not bundle a SoundFont.

## Comparable checkpoint samples

`scripts/sample_checkpoints.py` renders the same three prompts (piano in 4/4, bass in 4/4, and strings in 3/4) with fixed per-prompt RNG seeds. Each checkpoint gets its own directory containing canonical MIDI files and a JSON manifest with the source checkpoint, optimizer step, prompt tokens, generation settings, and artifact names. This makes audible changes between training milestones attributable to the checkpoint rather than to a changing prompt or random draw.

```bash
python scripts/sample_checkpoints.py \
  checkpoints/cadence.step-001000.pt checkpoints/cadence.step-002000.pt \
  --output-dir samples/checkpoints --tokens 1024
```

Add `--soundfont path/to/font.sf2` to render WAV files through FluidSynth. SoundFont distribution remains separate from Cadence.
