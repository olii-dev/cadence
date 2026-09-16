import json

import numpy as np

from cadence.model import ModelConfig
from cadence.train import train


def test_train_writes_loadable_checkpoint(tmp_path):
    data = np.tile(np.arange(31, dtype=np.uint16), 20)
    path = tmp_path / "tokens.npy"
    np.save(path, data)
    out = tmp_path / "smoke.pt"
    meta = train(
        path,
        out,
        ModelConfig(32, 16, 1, 2, 32, 0),
        steps=3,
        batch_size=2,
        learning_rate=0.003,
    )
    assert out.exists() and out.with_suffix(".json").exists()
    assert json.loads(out.with_suffix(".json").read_text())["steps"] == 3
    assert meta["final_loss"] < meta["initial_loss"]


def test_train_reads_merged_binary(tmp_path):
    data = np.tile(np.arange(31, dtype=np.uint16), 20)
    path = tmp_path / "tokens.bin"
    data.tofile(path)
    out = tmp_path / "binary-smoke.pt"
    meta = train(
        path,
        out,
        ModelConfig(32, 16, 1, 2, 32, 0),
        steps=3,
        batch_size=2,
        learning_rate=0.003,
    )
    assert out.exists() and meta["final_loss"] < meta["initial_loss"]


def test_accumulation_validation_scheduler_checkpoints_and_resume(tmp_path):
    data = np.tile(np.arange(31, dtype=np.uint16), 80)
    train_path = tmp_path / "train.bin"
    data.tofile(train_path)
    validation_path = tmp_path / "validation.bin"
    data[::-1].copy().tofile(validation_path)
    cfg = ModelConfig(32, 16, 1, 2, 32, 0)
    out = tmp_path / "run.pt"
    meta = train(
        train_path,
        out,
        cfg,
        steps=4,
        batch_size=2,
        learning_rate=0.004,
        accumulation_steps=2,
        min_learning_rate=0.001,
        warmup_steps=2,
        validation_data=validation_path,
        eval_interval=2,
        eval_batches=2,
        checkpoint_interval=2,
    )
    checkpoint = tmp_path / "run.step-000002.pt"
    assert checkpoint.exists() and checkpoint.with_suffix(".json").exists()
    assert meta["effective_batch_size"] == 4 and meta["validation_data"] == str(
        validation_path
    )
    assert [row["learning_rate"] for row in meta["history"][:2]] == [0.002, 0.004]
    assert [row["step"] for row in meta["history"] if "validation_loss" in row] == [
        1,
        2,
        4,
    ]
    resumed = tmp_path / "resumed.pt"
    resumed_meta = train(
        train_path,
        resumed,
        cfg,
        steps=5,
        batch_size=2,
        learning_rate=0.004,
        accumulation_steps=2,
        min_learning_rate=0.001,
        warmup_steps=2,
        validation_data=validation_path,
        eval_interval=2,
        eval_batches=2,
        resume=checkpoint,
    )
    assert resumed_meta["steps"] == 5
    assert [row["step"] for row in resumed_meta["history"]] == [1, 2, 3, 4, 5]
    state = __import__("torch").load(resumed, map_location="cpu", weights_only=True)
    assert {"model", "optimizer", "step", "train_generator_state"} <= state.keys()
