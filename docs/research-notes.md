# Research notes and sources

Implementation is original. These sources inform the representation and data choices; no repository code is copied.

- Lakh MIDI Dataset official page: https://colinraffel.com/projects/lmd/
  - 176,581 deduplicated MIDI files; 45,129 matched to Million Song Dataset entries.
  - The full archive intentionally retains a few thousand likely-corrupt files.
  - Curator distributes the collection as CC BY 4.0 and requests citation of Colin Raffel's 2016 thesis.
  - Important provenance caveat: the curator did not transcribe the songs; MIDI copyright metadata is inconsistent, so per-file attribution is often impossible.
- Pop Music Transformer / REMI paper: https://arxiv.org/abs/2002.00212
  - Beat/bar-aware events improve rhythmic structure over wall-clock performance events. Cadence follows the beat-aware principle, while designing its own multi-track representation and code.
- MidiTok paper (representation comparison, not code): https://hal.science/hal-03418930
- MidiTok tokenization survey/docs: https://miditok.readthedocs.io/en/latest/tokenizations.html
- MuseScore General soundfont license: https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General_License.md
  - Candidate free rendering dependency; bundle/distribution terms must be reviewed before shipping it with a demo. Users may point Cadence at any compatible local soundfont instead.
