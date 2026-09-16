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
