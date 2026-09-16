"""Kaggle entrypoint: fail closed without CUDA, then run a resumable config-driven job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cadence.model import ModelConfig
from cadence.sampling import sample_checkpoint
from cadence.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--validation-data", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--sample-dir", type=Path)
    p.add_argument("--sample-tokens", type=int, default=1024)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable: stop without training or consuming a CPU session"
        )
    raw = json.loads(a.config.read_text())
    model = ModelConfig(**raw["model"])
    t = raw["training"]
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "config": raw,
            },
            indent=2,
        ),
        flush=True,
    )

    def sample(path, step):
        if a.sample_dir:
            sample_checkpoint(
                path, a.sample_dir / f"step-{step:06d}", tokens=a.sample_tokens
            )

    meta = train(
        a.data,
        a.output,
        model,
        steps=t["steps"],
        batch_size=t["micro_batch_size"],
        accumulation_steps=t.get("gradient_accumulation", 1),
        learning_rate=t["learning_rate"],
        min_learning_rate=t.get("min_learning_rate", 0),
        warmup_steps=t.get("warmup_steps", 0),
        validation_data=a.validation_data,
        eval_interval=t.get("eval_interval", 0),
        eval_batches=t.get("eval_batches", 20),
        checkpoint_interval=t.get("checkpoint_interval", 0),
        seed=t["seed"],
        device="cuda",
        mixed_precision=True,
        resume=a.resume,
        checkpoint_callback=sample if a.sample_dir else None,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
