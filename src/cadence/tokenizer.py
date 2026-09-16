"""Original REMI-inspired MIDI tokenizer.

The representation is beat-aware but loss-aware: notes use quantized absolute positions,
program/channel identity, velocity, and duration. Tempo and meter are retained. Decoding
produces a canonical MIDI whose musical events round-trip exactly at the configured grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import mido

SPECIAL = ["PAD", "BOS", "EOS", "UNK"]


@dataclass(frozen=True)
class TokenizerConfig:
    positions_per_beat: int = 24
    velocity_bins: int = 32
    max_duration_beats: int = 32
    tempo_min: int = 30
    tempo_max: int = 240
    tempo_step: int = 2


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    values: tuple[int, ...]


class MidiTokenizer:
    """Converts MIDI to a deterministic event stream and back."""

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

    def _build_vocab(self) -> list[str]:
        c = self.config
        tokens = list(SPECIAL)
        tokens += [f"TIME_SHIFT_{i}" for i in range(1, c.positions_per_beat * 16 + 1)]
        tokens += [f"PITCH_{i}" for i in range(128)]
        tokens += [f"VELOCITY_{i}" for i in range(c.velocity_bins)]
        tokens += [f"DURATION_{i}" for i in range(1, c.positions_per_beat * c.max_duration_beats + 1)]
        tokens += [f"TRACK_{i}" for i in range(128)]
        tokens += [f"PROGRAM_{i}" for i in range(128)] + ["DRUMS"]
        tokens += [f"TEMPO_{i}" for i in range(c.tempo_min, c.tempo_max + 1, c.tempo_step)]
        tokens += [f"TIME_SIGNATURE_{n}_{d}" for d in (1, 2, 4, 8, 16) for n in range(1, 17)]
        return tokens

    def _grid_tick(self, tick: int, ticks_per_beat: int) -> int:
        return round(tick * self.config.positions_per_beat / ticks_per_beat)

    def _midi_tick(self, grid_tick: int, ticks_per_beat: int) -> int:
        return round(grid_tick * ticks_per_beat / self.config.positions_per_beat)

    def _velocity_bin(self, velocity: int) -> int:
        return min(self.config.velocity_bins - 1, velocity * self.config.velocity_bins // 128)

    def _velocity_value(self, bin_id: int) -> int:
        width = 128 / self.config.velocity_bins
        return max(1, min(127, round((bin_id + 0.5) * width)))

    def _tempo_bin(self, tempo_us: int) -> int:
        bpm = round(mido.tempo2bpm(tempo_us))
        c = self.config
        bpm = min(c.tempo_max, max(c.tempo_min, bpm))
        return c.tempo_min + round((bpm - c.tempo_min) / c.tempo_step) * c.tempo_step

    def extract_events(self, midi: mido.MidiFile) -> list[Event]:
        events: list[Event] = []
        voice_keys = []
        marked_tracks = {}
        for source_track, track in enumerate(midi.tracks):
            for message in track:
                if message.type == "track_name" and message.name.startswith("cadence_track_"):
                    try:
                        marked_tracks[source_track] = int(message.name.rsplit("_", 1)[1])
                    except ValueError:
                        pass
                if message.type in ("note_on", "note_off"):
                    key = (source_track, message.channel)
                    if key not in voice_keys:
                        voice_keys.append(key)
        canonical_voice = {source: index for index, source in enumerate(voice_keys)}
        for track_index, track in enumerate(midi.tracks):
            absolute = 0
            program_by_channel = {ch: 0 for ch in range(16)}
            open_notes: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            for msg in track:
                absolute += msg.time
                grid = self._grid_tick(absolute, midi.ticks_per_beat)
                if msg.type == "program_change":
                    program_by_channel[msg.channel] = msg.program
                elif msg.type == "set_tempo":
                    events.append(Event(grid, "tempo", (self._tempo_bin(msg.tempo),)))
                elif msg.type == "time_signature":
                    if msg.denominator in (1, 2, 4, 8, 16) and 1 <= msg.numerator <= 16:
                        events.append(Event(grid, "meter", (msg.numerator, msg.denominator)))
                elif msg.type == "note_on" and msg.velocity > 0:
                    key = (msg.channel, msg.note)
                    open_notes.setdefault(key, []).append(
                        (grid, self._velocity_bin(msg.velocity), program_by_channel[msg.channel])
                    )
                elif msg.type in ("note_off", "note_on"):
                    key = (msg.channel, msg.note)
                    if open_notes.get(key):
                        start, velocity, program = open_notes[key].pop(0)
                        duration = max(1, grid - start)
                        max_dur = self.config.positions_per_beat * self.config.max_duration_beats
                        events.append(Event(start, "note", (msg.note, velocity, min(duration, max_dur), program, msg.channel == 9, min(marked_tracks.get(track_index, canonical_voice.get((track_index, msg.channel), 0)), 127))))
        order = {"meter": 0, "tempo": 1, "note": 2}
        def sort_key(event: Event):
            if event.kind != "note":
                return (event.tick, order[event.kind], event.values)
            pitch, velocity, duration, program, drums, track = event.values
            return (event.tick, order[event.kind], track, drums, program, pitch, duration, velocity)
        return sorted(events, key=sort_key)

    def encode(self, midi_or_path: mido.MidiFile | str | Path) -> list[str]:
        midi = midi_or_path if isinstance(midi_or_path, mido.MidiFile) else mido.MidiFile(midi_or_path)
        tokens = ["BOS"]
        cursor = 0
        max_shift = self.config.positions_per_beat * 16
        for event in self.extract_events(midi):
            delta = event.tick - cursor
            while delta:
                shift = min(delta, max_shift)
                tokens.append(f"TIME_SHIFT_{shift}")
                cursor += shift
                delta -= shift
            if event.kind == "tempo":
                tokens.append(f"TEMPO_{event.values[0]}")
            elif event.kind == "meter":
                tokens.append(f"TIME_SIGNATURE_{event.values[0]}_{event.values[1]}")
            else:
                pitch, velocity, duration, program, drums, track_index = event.values
                tokens.extend([
                    f"TRACK_{track_index}", "DRUMS" if drums else f"PROGRAM_{program}",
                    f"PITCH_{pitch}", f"VELOCITY_{velocity}", f"DURATION_{duration}",
                ])
        tokens.append("EOS")
        return tokens

    def encode_ids(self, midi_or_path: mido.MidiFile | str | Path) -> list[int]:
        return [self.token_to_id[token] for token in self.encode(midi_or_path)]

    def decode(self, tokens: Iterable[str | int], ticks_per_beat: int = 480) -> mido.MidiFile:
        seq = [self.vocab[t] if isinstance(t, int) else t for t in tokens]
        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        meta_track = mido.MidiTrack(); midi.tracks.append(meta_track)
        instrument_tracks: dict[tuple[int, bool], list[tuple[int, mido.Message]]] = {}
        meta_events: list[tuple[int, mido.MetaMessage]] = []
        cursor = 0; current_program = 0; current_track = 0; drums = False; i = 0
        while i < len(seq):
            token = seq[i]
            if token.startswith("TIME_SHIFT_"):
                cursor += int(token.rsplit("_", 1)[1])
            elif token.startswith("TEMPO_"):
                bpm = int(token.rsplit("_", 1)[1])
                meta_events.append((cursor, mido.MetaMessage("set_tempo", tempo=round(mido.bpm2tempo(bpm)), time=0)))
            elif token.startswith("TIME_SIGNATURE_"):
                _, _, n, d = token.split("_")
                meta_events.append((cursor, mido.MetaMessage("time_signature", numerator=int(n), denominator=int(d), time=0)))
            elif token.startswith("TRACK_"):
                current_track = int(token.rsplit("_", 1)[1])
            elif token == "DRUMS":
                drums = True; current_program = 0
            elif token.startswith("PROGRAM_"):
                drums = False; current_program = int(token.rsplit("_", 1)[1])
            elif token.startswith("PITCH_") and i + 2 < len(seq):
                if seq[i + 1].startswith("VELOCITY_") and seq[i + 2].startswith("DURATION_"):
                    pitch = int(token.rsplit("_", 1)[1])
                    velocity = self._velocity_value(int(seq[i + 1].rsplit("_", 1)[1]))
                    duration = int(seq[i + 2].rsplit("_", 1)[1])
                    channel = 9 if drums else (current_track % 15)
                    if channel >= 9 and not drums: channel += 1
                    key = (current_track, current_program, drums)
                    track_events = instrument_tracks.setdefault(key, [])
                    track_events.append((cursor, mido.Message("note_on", channel=channel, note=pitch, velocity=velocity, time=0)))
                    track_events.append((cursor + duration, mido.Message("note_off", channel=channel, note=pitch, velocity=0, time=0)))
                    i += 2
            i += 1
        self._write_track(meta_track, meta_events, ticks_per_beat)
        for (track_index, program, is_drums), events in sorted(instrument_tracks.items()):
            track = mido.MidiTrack(); midi.tracks.append(track)
            channel = 9 if is_drums else (track_index * 7 + program) % 15
            if channel >= 9 and not is_drums: channel += 1
            track.append(mido.MetaMessage("track_name", name=f"cadence_track_{track_index}", time=0))
            if not is_drums:
                track.append(mido.Message("program_change", channel=channel, program=program, time=0))
            for _, message in events:
                message.channel = channel
            event_order = {"program_change": 0, "note_off": 1, "note_on": 2}
            self._write_track(track, sorted(events, key=lambda x: (x[0], event_order[x[1].type], getattr(x[1], "note", -1))), ticks_per_beat)
        return midi

    def _write_track(self, track, events, ticks_per_beat: int) -> None:
        previous = 0
        for grid, msg in sorted(events, key=lambda x: x[0]):
            tick = self._midi_tick(grid, ticks_per_beat)
            msg.time = max(0, tick - previous); previous = tick; track.append(msg)
        track.append(mido.MetaMessage("end_of_track", time=0))

    def save_vocab(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"config": self.config.__dict__, "vocab": self.vocab}, indent=2))
