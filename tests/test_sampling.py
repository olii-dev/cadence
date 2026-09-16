import numpy as np

from cadence.model import ModelConfig
from cadence.sampling import sample_checkpoint
from cadence.train import train


def test_fixed_checkpoint_suite_is_reproducible_and_valid(tmp_path):
    data = np.tile(np.arange(31, dtype=np.uint16), 30)
    data_path = tmp_path / "tokens.bin"
    data.tofile(data_path)
    checkpoint = tmp_path / "tiny.pt"
    train(
        data_path, checkpoint, ModelConfig(1759, 16, 1, 2, 32, 0), steps=1, batch_size=2
    )
    prompts = {"one": ["BOS", "TRACK_0", "PROGRAM_0"]}
    first = sample_checkpoint(
        checkpoint, tmp_path / "first", tokens=12, prompts=prompts
    )
    sample_checkpoint(checkpoint, tmp_path / "second", tokens=12, prompts=prompts)
    first_midi = (tmp_path / "first" / "one.mid").read_bytes()
    second_midi = (tmp_path / "second" / "one.mid").read_bytes()
    assert first_midi == second_midi
    assert first["samples"][0]["generated_tokens"] == 12
    assert first["seed"] == 1729 and (tmp_path / "first" / "manifest.json").exists()
