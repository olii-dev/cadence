from pathlib import Path
import json
import mido
import numpy as np
from cadence.dataset import build_dataset


def write_song(path: Path, pitch: int):
    midi=mido.MidiFile(ticks_per_beat=480); track=mido.MidiTrack(); midi.tracks.append(track)
    for i in range(16):
        track.append(mido.Message('note_on',note=pitch+(i%3),velocity=80,time=0 if i==0 else 120))
        track.append(mido.Message('note_off',note=pitch+(i%3),velocity=0,time=120))
    midi.save(path)


def test_resumable_range_and_outputs(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    for i in range(3): write_song(source/f'{i}.mid',60+i)
    out=tmp_path/'out'; stats=build_dataset(source,out,start=1,limit=1)
    summary=json.loads((out/'stats.json').read_text())
    assert stats.discovered == 1
    assert summary['source_total'] == 3
    assert summary['range_start'] == 1 and summary['range_end'] == 2
    assert sum(len((out/f'{split}.jsonl').read_text().splitlines()) for split in ('train','validation','test')) == 1
    assert sum(np.load(out/f'{split}.npy').size for split in ('train','validation','test')) > 0
