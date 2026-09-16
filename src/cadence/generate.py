"""Generate MIDI and optionally render it through FluidSynth."""
import argparse, json, subprocess
from pathlib import Path
import torch
from .model import ModelConfig, MusicTransformer
from .tokenizer import MidiTokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--tokens',type=int,default=1024); p.add_argument('--temperature',type=float,default=.95); p.add_argument('--top-k',type=int,default=50)
    p.add_argument('--soundfont',type=Path); a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); cfg=ModelConfig(**state['config']); model=MusicTransformer(cfg); model.load_state_dict(state['model']); model.eval()
    tok=MidiTokenizer(); start=torch.tensor([[tok.token_to_id['BOS']]])
    def grammar(prefix, logits):
        allowed = tok.allowed_next_ids(prefix[0].tolist())
        mask = torch.full_like(logits, float("-inf")); mask[:, allowed] = logits[:, allowed]
        return mask
    ids=model.generate(start,a.tokens,a.temperature,a.top_k,tok.token_to_id['EOS'],logits_processor=grammar)[0].tolist()
    tok.decode(ids).save(a.output)
    if a.soundfont:
        wav=a.output.with_suffix('.wav'); subprocess.run(['fluidsynth','-ni',str(a.soundfont),str(a.output),'-F',str(wav),'-r','44100'],check=True)
        print(wav)
    print(a.output)
if __name__=='__main__': main()
