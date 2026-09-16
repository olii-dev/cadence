"""Generate a comparable fixed-prompt suite from one or more checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from cadence.sampling import sample_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--soundfont", type=Path)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        destination = args.output_dir / checkpoint.stem
        sample_checkpoint(
            checkpoint,
            destination,
            seed=args.seed,
            tokens=args.tokens,
            soundfont=args.soundfont,
        )
        print(destination)


if __name__ == "__main__":
    main()
