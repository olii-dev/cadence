# Kaggle path

Cadence's Kaggle path within the Lattice model family is designed for one free T4/P100 and for resumable, economical use of Oli's shared weekly quota.

1. Upload or clone this repository in a Kaggle notebook.
2. Enable internet for the preparation phase. Run `uv sync`, then `scripts/kaggle_prepare.sh`.
3. The script downloads the official Lakh archive, checks the exact SHA-256, verifies 178,561 physical MIDI files, preprocesses resumable 5,000-file shards, and globally deduplicates them.
4. Keep `/kaggle/working/cadence-data/merged` as a Kaggle output dataset so later GPU sessions do not repeat CPU preprocessing or redownload 1.65 GiB.
5. Start the small run from `configs/kaggle-small.json`. It is an 8-layer, width-512 GPT with 1,024-token context, about 27M parameters, effective batch 32 through gradient accumulation, and mixed precision on CUDA.

The first run should stop at an evaluation checkpoint if validation loss stops improving. Do not spend the whole weekly quota merely because it exists. The first target is a defensible quality curve and several held-out samples; scale only when those support it.
