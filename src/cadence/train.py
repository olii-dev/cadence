"""Train Cadence checkpoints from packed uint16 token shards."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import json, time
import numpy as np
import torch
from .model import ModelConfig, MusicTransformer


def batches(tokens, batch_size, context, device, generator):
    high = len(tokens) - context - 1
    if high <= 0: raise ValueError("dataset is shorter than one context window")
    while True:
        offsets = torch.randint(high, (batch_size,), generator=generator).tolist()
        x = torch.stack([torch.from_numpy(tokens[i:i+context].astype(np.int64)) for i in offsets]).to(device)
        y = torch.stack([torch.from_numpy(tokens[i+1:i+context+1].astype(np.int64)) for i in offsets]).to(device)
        yield x, y


def open_tokens(data: Path):
    if data.suffix == ".npy":
        return np.load(data, mmap_mode="r")
    if data.suffix == ".bin":
        return np.memmap(data, dtype=np.uint16, mode="r")
    raise ValueError("training data must be a .npy or raw uint16 .bin file")


def train(data: Path, output: Path, cfg: ModelConfig, steps=1000, batch_size=8, learning_rate=3e-4, seed=1729, device="cpu"):
    torch.manual_seed(seed); generator=torch.Generator().manual_seed(seed)
    tokens=open_tokens(data); stream=batches(tokens,batch_size,cfg.context_length,device,generator)
    model=MusicTransformer(cfg).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=(.9,.95),weight_decay=.1)
    losses=[]; started=time.time(); model.train()
    for step in range(1,steps+1):
        x,y=next(stream); _,loss=model(x,y); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item())
        if step == 1 or step % max(1,steps//10) == 0: print(f"step={step} loss={loss.item():.4f}",flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    metadata={"config":asdict(cfg),"steps":steps,"batch_size":batch_size,"learning_rate":learning_rate,"seed":seed,"data":str(data),"parameters":sum(p.numel() for p in model.parameters()),"initial_loss":losses[0],"final_loss":losses[-1],"min_loss":min(losses),"elapsed_sec":time.time()-started}
    torch.save({"model":model.state_dict(),"config":asdict(cfg),"metadata":metadata},output)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)); return metadata


def main():
    p=argparse.ArgumentParser(); p.add_argument('data',type=Path); p.add_argument('output',type=Path); p.add_argument('--vocab-size',type=int,required=True)
    p.add_argument('--steps',type=int,default=1000); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--context',type=int,default=1024)
    p.add_argument('--layers',type=int,default=8); p.add_argument('--heads',type=int,default=8); p.add_argument('--width',type=int,default=512); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--learning-rate',type=float,default=3e-4); p.add_argument('--device',default='cpu')
    a=p.parse_args(); cfg=ModelConfig(a.vocab_size,a.context,a.layers,a.heads,a.width,a.dropout); print(json.dumps(train(a.data,a.output,cfg,a.steps,a.batch_size,a.learning_rate,device=a.device),indent=2))
if __name__=='__main__': main()
