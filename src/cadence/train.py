"""Train Cadence checkpoints from packed uint16 token shards."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, MusicTransformer


def batches(tokens, batch_size, context, device, generator):
    high = len(tokens) - context - 1
    if high <= 0:
        raise ValueError("dataset is shorter than one context window")
    while True:
        offsets = torch.randint(high, (batch_size,), generator=generator).tolist()
        x = torch.stack(
            [
                torch.from_numpy(tokens[i : i + context].astype(np.int64))
                for i in offsets
            ]
        ).to(device)
        y = torch.stack(
            [
                torch.from_numpy(tokens[i + 1 : i + context + 1].astype(np.int64))
                for i in offsets
            ]
        ).to(device)
        yield x, y


def open_tokens(data: Path):
    if data.suffix == ".npy":
        return np.load(data, mmap_mode="r")
    if data.suffix == ".bin":
        return np.memmap(data, dtype=np.uint16, mode="r")
    raise ValueError("training data must be a .npy or raw uint16 .bin file")


def scheduled_learning_rate(step, *, peak, minimum, warmup, total):
    """Linear warmup followed by cosine decay, indexed by optimizer step."""
    if warmup and step <= warmup:
        return peak * step / warmup
    if total <= warmup:
        return minimum
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, stream, batches_count, autocast):
    model.eval()
    losses = []
    for _ in range(batches_count):
        x, y = next(stream)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _checkpoint_path(output: Path, step: int):
    return output.with_name(f"{output.stem}.step-{step:06d}{output.suffix}")


def train(
    data: Path,
    output: Path,
    cfg: ModelConfig,
    steps=1000,
    batch_size=8,
    learning_rate=3e-4,
    seed=1729,
    device="cpu",
    *,
    accumulation_steps=1,
    min_learning_rate=0.0,
    warmup_steps=0,
    validation_data: Path | None = None,
    eval_interval=0,
    eval_batches=20,
    checkpoint_interval=0,
    mixed_precision=False,
    resume: Path | None = None,
):
    if steps < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("steps, batch_size, and accumulation_steps must be positive")
    torch.manual_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    tokens = open_tokens(data)
    stream = batches(tokens, batch_size, cfg.context_length, device, train_generator)
    validation_stream = None
    if validation_data:
        validation_stream = batches(
            open_tokens(validation_data),
            batch_size,
            cfg.context_length,
            device,
            validation_generator,
        )
    model = MusicTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    use_amp = mixed_precision and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else nullcontext
    )
    start_step = 0
    history = []
    initial_loss = None
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["config"] != asdict(cfg):
            raise ValueError("resume checkpoint model config does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if use_amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        history = list(state.get("history", []))
        if history:
            initial_loss = history[0]["train_loss"]
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda_rng_state_all"]]
            )
        if state.get("train_generator_state") is not None:
            train_generator.set_state(state["train_generator_state"].cpu())
        if state.get("validation_generator_state") is not None:
            validation_generator.set_state(state["validation_generator_state"].cpu())
        print(f"resumed={resume} step={start_step}", flush=True)
    if start_step >= steps:
        raise ValueError("resume checkpoint already reached requested steps")
    started = time.time()
    model.train()

    def save(path, step, final=False):
        metadata = {
            "config": asdict(cfg),
            "steps": step,
            "target_steps": steps,
            "batch_size": batch_size,
            "gradient_accumulation": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": warmup_steps,
            "seed": seed,
            "data": str(data),
            "validation_data": str(validation_data) if validation_data else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_loss": initial_loss,
            "final_loss": history[-1]["train_loss"],
            "min_loss": min(h["train_loss"] for h in history),
            "elapsed_sec": time.time() - started,
            "history": history,
            "mixed_precision": use_amp,
            "resumed_from": str(resume) if resume else None,
        }
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "config": asdict(cfg),
            "step": step,
            "history": history,
            "train_generator_state": train_generator.get_state(),
            "validation_generator_state": validation_generator.get_state(),
            "metadata": metadata,
        }
        state["torch_rng_state"] = torch.get_rng_state()
        state["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        metadata_path = path.with_suffix(".json")
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(json.dumps(metadata, indent=2))
        metadata_temporary.replace(metadata_path)
        if not final:
            print(f"checkpoint={path}", flush=True)
        return metadata

    for step in range(start_step + 1, steps + 1):
        lr = scheduled_learning_rate(
            step,
            peak=learning_rate,
            minimum=min_learning_rate,
            warmup=warmup_steps,
            total=steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(accumulation_steps):
            x, y = next(stream)
            with autocast():
                _, loss = model(x, y)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            micro_losses.append(loss.item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss = sum(micro_losses) / len(micro_losses)
        if initial_loss is None:
            initial_loss = train_loss
        record = {"step": step, "train_loss": train_loss, "learning_rate": lr}
        if (
            validation_stream
            and eval_interval
            and (step == 1 or step % eval_interval == 0 or step == steps)
        ):
            record["validation_loss"] = evaluate(
                model, validation_stream, eval_batches, autocast
            )
        history.append(record)
        if step == 1 or step % max(1, steps // 10) == 0 or "validation_loss" in record:
            suffix = (
                f" validation_loss={record['validation_loss']:.4f}"
                if "validation_loss" in record
                else ""
            )
            print(f"step={step} loss={train_loss:.4f} lr={lr:.6g}{suffix}", flush=True)
        if checkpoint_interval and step % checkpoint_interval == 0 and step < steps:
            save(_checkpoint_path(output, step), step)
    return save(output, steps, final=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    cfg = ModelConfig(a.vocab_size, a.context, a.layers, a.heads, a.width, a.dropout)
    print(
        json.dumps(
            train(
                a.data,
                a.output,
                cfg,
                a.steps,
                a.batch_size,
                a.learning_rate,
                device=a.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
