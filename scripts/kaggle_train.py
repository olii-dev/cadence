"""Kaggle entrypoint: smoke-gate CUDA, then launch a config-driven Cadence run."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from cadence.model import ModelConfig
from cadence.train import train

def main():
    p=argparse.ArgumentParser(); p.add_argument('data',type=Path); p.add_argument('config',type=Path); p.add_argument('output',type=Path); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable: stop without training or consuming a CPU session')
    raw=json.loads(a.config.read_text()); model=ModelConfig(**raw['model']); t=raw['training']
    print(json.dumps({'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'config':raw},indent=2),flush=True)
    # Current trainer deliberately maps only supported knobs. Accumulation/scheduler/eval are the next gated training-loop additions.
    meta=train(a.data,a.output,model,steps=t['steps'],batch_size=t['micro_batch_size'],learning_rate=t['learning_rate'],seed=t['seed'],device='cuda')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
