"""KV-cache generation must reproduce the original full-recompute path exactly."""
import torch
import torch.nn.functional as F
from cadence.model import ModelConfig, MusicTransformer


def reference_generate(model, ids, max_new_tokens, temperature=1.0, top_k=50, eos_id=None, logits_processor=None):
    """The pre-KV-cache generate loop, kept here as the golden reference."""
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.context_length:])
        logits = logits[:, -1] / max(temperature, 1e-5)
        if logits_processor is not None:
            logits = logits_processor(ids, logits)
        if top_k:
            logits[logits < torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]] = float('-inf')
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat((ids, next_id), dim=1)
        if eos_id is not None and torch.all(next_id == eos_id):
            break
    return ids


def make_model(vocab=64, ctx=32, layers=2, heads=2, width=32):
    torch.manual_seed(1234)
    model = MusicTransformer(ModelConfig(vocab_size=vocab, context_length=ctx, layers=layers, heads=heads, width=width, dropout=0))
    return model.eval()


def test_cached_forward_matches_full_forward():
    model = make_model()
    ids = torch.randint(0, 64, (2, 10))
    full_logits, _ = model(ids)
    prefill_logits, _, caches = model(ids[:, :6], use_cache=True)
    assert torch.allclose(full_logits[:, :6], prefill_logits, atol=1e-5)
    step_logits, _, caches = model(ids[:, 6:7], caches=caches, use_cache=True)
    assert torch.allclose(full_logits[:, 6:7], step_logits, atol=1e-5)
    rest_logits, _, _ = model(ids[:, 7:], caches=caches, use_cache=True)
    assert torch.allclose(full_logits[:, 7:], rest_logits, atol=1e-5)


def test_generate_matches_reference_greedy():
    model = make_model()
    ids = torch.tensor([[1, 5, 9]])
    expected = reference_generate(model, ids, max_new_tokens=24, top_k=1)
    actual = model.generate(ids, max_new_tokens=24, top_k=1)
    assert torch.equal(expected, actual)


def test_generate_matches_reference_seeded_sampling():
    model = make_model()
    ids = torch.tensor([[1, 5, 9]])
    torch.manual_seed(99)
    expected = reference_generate(model, ids, max_new_tokens=24, temperature=0.9, top_k=12)
    torch.manual_seed(99)
    actual = model.generate(ids, max_new_tokens=24, temperature=0.9, top_k=12)
    assert torch.equal(expected, actual)


def test_generate_with_logits_processor_matches_reference():
    model = make_model()
    ids = torch.tensor([[1, 5, 9]])

    def allow_even(prefix, logits):
        mask = torch.full_like(logits, float('-inf'))
        mask[:, 0::2] = logits[:, 0::2]
        return mask

    torch.manual_seed(7)
    expected = reference_generate(model, ids, max_new_tokens=16, temperature=0.95, top_k=50, logits_processor=allow_even)
    torch.manual_seed(7)
    actual = model.generate(ids, max_new_tokens=16, temperature=0.95, top_k=50, logits_processor=allow_even)
    assert torch.equal(expected, actual)
    assert all(token % 2 == 0 for token in actual[0, 3:].tolist())


def test_generate_respects_eos():
    model = make_model()
    ids = torch.tensor([[1, 5, 9]])
    torch.manual_seed(3)
    expected = reference_generate(model, ids, max_new_tokens=24, top_k=1, eos_id=2)
    torch.manual_seed(3)
    actual = model.generate(ids, max_new_tokens=24, top_k=1, eos_id=2)
    assert torch.equal(expected, actual)


def test_generate_past_context_falls_back_to_window():
    model = make_model(ctx=12)
    ids = torch.tensor([[1, 5, 9]])
    torch.manual_seed(11)
    expected = reference_generate(model, ids, max_new_tokens=20, top_k=1)
    torch.manual_seed(11)
    actual = model.generate(ids, max_new_tokens=20, top_k=1)
    assert torch.equal(expected, actual)
    assert actual.shape == (1, 23)


def test_generate_clips_long_prompt():
    model = make_model(ctx=8)
    ids = torch.arange(20).reshape(1, 20) % 64
    actual = model.generate(ids, max_new_tokens=2, top_k=1)
    assert actual.shape[1] == 22
