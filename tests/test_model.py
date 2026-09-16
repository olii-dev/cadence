import torch
from cadence.model import ModelConfig, MusicTransformer

def test_model_forward_and_generation():
    cfg=ModelConfig(vocab_size=32,context_length=16,layers=2,heads=2,width=32,dropout=0)
    model=MusicTransformer(cfg); ids=torch.randint(0,32,(2,16)); logits,loss=model(ids,ids)
    assert logits.shape==(2,16,32) and loss.item()>0
    generated=model.generate(ids[:1,:4],max_new_tokens=4,top_k=5)
    assert generated.shape==(1,8)
