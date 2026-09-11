#!/usr/bin/env python3
"""PRISM P3 — 门来源 x 值通路 2x2 分解（正确版：PCN 内部开关）"""
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
    m = torch.zeros(B, seq.size(1) - 1)
    m[:, H:] = 1.0
    return seq[:, :-1].to(DEV), seq[:, 1:].to(DEV), m.to(DEV)

@torch.no_grad()
def copy_eval(m):
    tots, ns = [], []
    for i in range(3):
        x, y, mm = gen(16, seed=7000 + i)
        l = F.cross_entropy(m(x, topk=16).reshape(-1, V), y.reshape(-1), reduction='none')
        cl = l[mm.reshape(-1) > 0.5]
        tots.append(cl.sum().item()); ns.append(cl.numel())
    return sum(tots) / max(sum(ns), 1)

def run(gs, vs, seed=0):
    torch.manual_seed(seed)
    m = PCNModel(vocab_size=V, d_model=D, n_layers=L, n_heads=HEADS,
                 d_gate=DG, dropout=0.0, max_seq_len=2 * H + 1,
                 init_mode='fixed', gate_source=gs, val_source=vs).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(STEPS):
        x, y, _ = gen(32)
        loss = F.cross_entropy(m(x, topk=16).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return round(copy_eval(m), 3)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | PCN internal-switch 2x2 | 3 seed")
    print("===== calibration: feat/wlat_e (native, known GOOD) =====")
    cal = [run('feat', 'wlat_e', seed=s) for s in range(3)]
    ok = sum(1 for c in cal if c < 1.0)
    print(f"  3 seed: {cal}  learned {ok}/3")
    if ok < 2:
        json.dump({'calibration': cal, 'valid': False,
                   'note': 'control cell failed (<2/3) - experiment void'},
                  open(OUT / 'gate_source_v2.json', 'w'), indent=2)
        print("  CALIBRATION FAIL - skip experiment cells")
        return
    results = [{'name': 'feat/wlat_e(control)', 'copies': cal, 'mean': round(sum(cal)/3, 3)}]
    for name, gs, vs in (('qk/wlat_e', 'qk', 'wlat_e'), ('feat/wlat_h', 'feat', 'wlat_h'), ('qk/wlat_h', 'qk', 'wlat_h')):
        copies = [run(gs, vs, seed=s) for s in range(3)]
        okn = sum(1 for c in copies if c < 1.0)
        print(f"  {name:<18} 3 seed: {copies}  learned {okn}/3")
        results.append({'name': name, 'copies': copies, 'mean': round(sum(copies)/3, 3), 'learned': f'{okn}/3'})
    json.dump({'calibration': cal, 'valid': True, 'results': results},
              open(OUT / 'gate_source_v2.json', 'w'), indent=2)
    print(f"saved: {OUT}/gate_source_v2.json")

if __name__ == '__main__':
    main()
