"""A compact decoder-only transformer, implemented from first principles in PyTorch."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 1024
    layers: int = 8
    heads: int = 8
    width: int = 512
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); assert cfg.width % cfg.heads == 0
        self.heads = cfg.heads; self.head_dim = cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False); self.dropout = cfg.dropout
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b,t,self.heads,self.head_dim).transpose(1,2)
        k = k.view(b,t,self.heads,self.head_dim).transpose(1,2)
        v = v.view(b,t,self.heads,self.head_dim).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.ln1=nn.LayerNorm(cfg.width); self.attn=CausalSelfAttention(cfg); self.ln2=nn.LayerNorm(cfg.width)
        self.mlp=nn.Sequential(nn.Linear(cfg.width,4*cfg.width),nn.GELU(),nn.Linear(4*cfg.width,cfg.width),nn.Dropout(cfg.dropout))
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MusicTransformer(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__(); self.cfg=cfg
        self.token=nn.Embedding(cfg.vocab_size,cfg.width); self.position=nn.Embedding(cfg.context_length,cfg.width)
        self.drop=nn.Dropout(cfg.dropout); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.layers)]); self.norm=nn.LayerNorm(cfg.width)
        self.head=nn.Linear(cfg.width,cfg.vocab_size,bias=False); self.head.weight=self.token.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0,std=.02)
        if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None):
        _,t=ids.shape
        if t>self.cfg.context_length: raise ValueError("sequence exceeds context length")
        x=self.drop(self.token(ids)+self.position(torch.arange(t,device=ids.device)))
        for block in self.blocks: x=block(x)
        logits=self.head(self.norm(x)); loss=None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten())
        return logits,loss
    @torch.no_grad()
    def generate(self,ids,max_new_tokens=512,temperature=1.0,top_k=50,eos_id=None,logits_processor=None):
        for _ in range(max_new_tokens):
            logits,_=self(ids[:,-self.cfg.context_length:]); logits=logits[:,-1]/max(temperature,1e-5)
            if logits_processor is not None:
                logits = logits_processor(ids, logits)
            if top_k: logits[logits < torch.topk(logits,min(top_k,logits.size(-1))).values[:,-1,None]]=float('-inf')
            next_id=torch.multinomial(F.softmax(logits,dim=-1),1); ids=torch.cat((ids,next_id),dim=1)
            if eos_id is not None and torch.all(next_id==eos_id): break
        return ids
