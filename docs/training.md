# Training and export

`cadence-train` consumes a packed `uint16` NumPy shard, samples deterministic context windows, trains the original decoder-only transformer with AdamW, clips gradients, and writes:

- a PyTorch checkpoint with model weights, full architecture config, and run metadata
- a sidecar JSON record with data path, seed, steps, batch size, learning rate, parameter count, loss, and elapsed time

This explicit config-plus-state-dict format is small, inspectable, and easy to wrap for Hugging Face and Lattice serving later without making either service a training dependency.

## Real-data CPU smoke run

The first smoke run used 9.7M real Lakh tokens from the initial corpus shard. A 637,312-parameter model (2 layers, width 128, 4 heads, context 128) trained for 80 CPU steps:

- initial loss: 7.5007
- final loss: 3.8793
- minimum loss: 3.5335
- model-loop elapsed time: 3.40 seconds

Sampling 512 new tokens from that checkpoint produced a valid 24-track MIDI with 61 notes and 3.19 seconds of playback. This confirms the real-token checkpoint-to-generation pipeline. It is intentionally undertrained and not a quality result.
