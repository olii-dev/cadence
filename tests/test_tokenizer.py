import mido
from cadence.tokenizer import MidiTokenizer


def sample_midi():
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack(); midi.tracks.append(meta)
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track = mido.MidiTrack(); midi.tracks.append(track)
    track.append(mido.Message("program_change", program=4, channel=0, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    track.append(mido.Message("note_on", note=64, velocity=96, channel=0, time=240))
    track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=960))
    drums = mido.MidiTrack(); midi.tracks.append(drums)
    drums.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
    drums.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=120))
    return midi


def test_token_round_trip_is_canonical():
    tokenizer = MidiTokenizer()
    first = tokenizer.encode(sample_midi())
    second = tokenizer.encode(tokenizer.decode(first))
    assert first == second


def test_id_round_trip():
    tokenizer = MidiTokenizer()
    ids = tokenizer.encode_ids(sample_midi())
    assert tokenizer.encode(tokenizer.decode(ids)) == tokenizer.encode(sample_midi())


def test_long_time_shift_and_overlapping_same_pitch_survive():
    midi = mido.MidiFile(ticks_per_beat=480); track = mido.MidiTrack(); midi.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=8000))
    track.append(mido.Message("note_on", note=60, velocity=100, time=120))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    tok = MidiTokenizer(); encoded = tok.encode(midi)
    assert encoded.count("PITCH_60") == 2
    assert tok.encode(tok.decode(encoded)) == encoded
