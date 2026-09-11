#!/usr/bin/env python3
"""R3 — 2x2x2 因子设计补格（W3 审稿建议）：门来源 x 值对象 x FFN
已有格（3 seed）: feat/e(-FFN)=0.21, qk/e=0.09, feat/h=0.022, qk/h=0.047, qk/h+FFN=0.074
补缺格: feat/e+FFN, qk/e+FFN, feat/h+FFN（各 3 seed）
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, SEP, H = 32, 31, 24
D, L, HEADS, DG = 64, 12, 4, 16
STEPS = 1000
OUT = Path('results/mechanism')

def gen(B, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    kw = dict(generator=g) if g is not None else {}
    head = torch.randint(0, SEP, (B, H), **kw)
    seq = torch.cat([head, torch.full((B, 1), SEP, dtype=torch.long), head], dim=1)
    m = torch.zeros(B, seq.size(1) - 1); m[:, H:] = 1.0
    return seq[:, :-1].to(DEV), seq[:, 1:].to(DEV), m.to(DEV)

@torch.no_grad()
def copy_eval(m):
    tots, ns = [], []
    for i in range(3):
        x, y, mm = gen(16, seed=7000+i)
        l = F.cross_entropy(m(x, topk=16).reshape(-1, V), y.reshape(-1), reduction='none')
        cl = l[mm.reshape(-1) > 0.5]; tots.append(cl.sum().item()); ns.append(cl.numel())
    return sum(tots)/max(sum(ns), 1)

def run(gs, vs, ffn, seed=0):
    torch.manual_seed(seed)
    m = PCNModel(vocab_size=V, d_model=D, n_layers=L, n_heads=HEADS, d_gate=DG,
                 dropout=0.0, max_seq_len=2*H+1, init_mode='fixed',
                 gate_source=gs, val_source=vs, use_ffn=ffn).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(STEPS):
        x, y, _ = gen(32)
        loss = F.cross_entropy(m(x, topk=16).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return round(copy_eval(m), 3)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('R3 factorial fill: 3 missing cells x 3 seeds (CPU)')
    results = {}
    for name, gs, vs, ffn in (('feat_e_ffn', 'feat', 'wlat_e', True),
                               ('qk_e_ffn', 'qk', 'wlat_e', True),
                               ('feat_h_ffn', 'feat', 'wlat_h', True)):
        copies = [run(gs, vs, ffn, seed=s) for s in range(3)]
        ok = sum(1 for c in copies if c < 1.0)
        print(f'  {name:<14} {copies}  learned {ok}/3')
        results[name] = copies
    json.dump(results, open(OUT/'factorial_fill.json', 'w'), indent=2)
    print(f'saved {OUT}/factorial_fill.json')

if __name__ == '__main__':
    main()
