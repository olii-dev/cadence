import json
from pathlib import Path

def test_smoke_precedes_full_run_and_has_bounded_steps():
    smoke=json.loads(Path('configs/kaggle-smoke.json').read_text()); full=json.loads(Path('configs/kaggle-small.json').read_text())
    assert smoke['training']['steps'] <= 500
    assert smoke['model']['width'] < full['model']['width']
    assert smoke['model']['layers'] < full['model']['layers']
