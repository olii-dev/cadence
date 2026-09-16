"""Overfit a synthetic musical phrase to prove the training loop."""
import argparse, json, torch
from cadence.model import ModelConfig, MusicTransformer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=120); a=p.parse_args(); torch.manual_seed(1729)
    phrase=torch.tensor([1,10,20,30,20,40,10,20,30,20,40,2]*8,dtype=torch.long)
    x=phrase[:-1].unsqueeze(0); y=phrase[1:].unsqueeze(0)
    cfg=ModelConfig(vocab_size=64,context_length=len(x[0]),layers=2,heads=2,width=64,dropout=0)
    model=MusicTransformer(cfg); opt=torch.optim.AdamW(model.parameters(),lr=3e-3); losses=[]
    for step in range(a.steps):
        _,loss=model(x,y); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    with torch.no_grad(): accuracy=(model(x)[0].argmax(-1)==y).float().mean().item()
    print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'initial_loss':losses[0],'final_loss':losses[-1],'token_accuracy':accuracy,'steps':a.steps},indent=2))
if __name__=='__main__': main()
