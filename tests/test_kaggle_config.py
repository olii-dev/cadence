import json
from pathlib import Path
from cadence.model import ModelConfig, MusicTransformer

def test_kaggle_config_is_valid_and_small():
    cfg=json.loads(Path('configs/kaggle-small.json').read_text())
    model=MusicTransformer(ModelConfig(**cfg['model']))
    params=sum(p.numel() for p in model.parameters())
    assert 20_000_000 < params < 40_000_000
    assert cfg['training']['micro_batch_size'] * cfg['training']['gradient_accumulation'] == 32
    assert cfg['model']['context_length'] == 1024
