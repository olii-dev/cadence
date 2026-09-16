"""Deterministic checkpoint sampling for comparable training milestones."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from .model import ModelConfig, MusicTransformer
from .tokenizer import MidiTokenizer

DEFAULT_PROMPTS = {
    "piano": ["BOS", "TEMPO_120", "TIME_SIGNATURE_4_4", "TRACK_0", "PROGRAM_0"],
    "bass_drums": ["BOS", "TEMPO_100", "TIME_SIGNATURE_4_4", "TRACK_0", "PROGRAM_32"],
    "strings": ["BOS", "TEMPO_80", "TIME_SIGNATURE_3_4", "TRACK_0", "PROGRAM_48"],
}


def sample_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    seed: int = 1729,
    tokens: int = 1024,
    temperature: float = 0.95,
    top_k: int = 50,
    soundfont: Path | None = None,
    prompts: dict[str, list[str]] | None = None,
):
    """Render a fixed prompt suite to MIDI and optional WAV with a fixed RNG seed."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**state["config"])
    model = MusicTransformer(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    tokenizer = MidiTokenizer()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(checkpoint),
        "step": state.get("step") or state.get("metadata", {}).get("steps"),
        "seed": seed,
        "tokens": tokens,
        "temperature": temperature,
        "top_k": top_k,
        "samples": [],
    }
    for index, (name, prompt) in enumerate((prompts or DEFAULT_PROMPTS).items()):
        unknown = [token for token in prompt if token not in tokenizer.token_to_id]
        if unknown:
            raise ValueError(f"unknown prompt tokens: {unknown}")
        prompt_ids = [tokenizer.token_to_id[token] for token in prompt]
        if any(token_id >= cfg.vocab_size for token_id in prompt_ids):
            raise ValueError(
                "checkpoint vocabulary is incompatible with the MIDI tokenizer"
            )
        torch.manual_seed(seed + index)
        prefix = torch.tensor([prompt_ids])

        def grammar(generated, logits):
            allowed = tokenizer.allowed_next_ids(generated[0].tolist())
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed] = logits[:, allowed]
            return mask

        ids = model.generate(
            prefix,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.token_to_id["EOS"],
            logits_processor=grammar,
        )[0].tolist()
        midi_path = output_dir / f"{name}.mid"
        tokenizer.decode(ids).save(midi_path)
        record = {
            "name": name,
            "prompt": prompt,
            "seed": seed + index,
            "midi": midi_path.name,
            "generated_tokens": len(ids) - len(prompt),
        }
        if soundfont:
            wav_path = output_dir / f"{name}.wav"
            subprocess.run(
                [
                    "fluidsynth",
                    "-ni",
                    str(soundfont),
                    str(midi_path),
                    "-F",
                    str(wav_path),
                    "-r",
                    "44100",
                ],
                check=True,
            )
            record["audio"] = wav_path.name
        manifest["samples"].append(record)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
