#!/usr/bin/env python3
"""PRISM C-plan — 块结构子件二分：FFN 干扰假说检验

已知锚点（对照格）：
  tf_sigmoid（qk门+V值+标准块含FFN）= 3.435 BAD  [copy_mechanism]
  PCN qk/h（qk门+W_lat(h)值+PCN块无FFN）= 0.047 GOOD  [gate_source_v2]
假说：FFN 的逐位置通道混合稀释位置 token 身份 → 干扰精确复制

加法线（标准块逐件 PCN 化，S0 必须复现 BAD）：
  S1: 去 FFN
  S2: 更新通路（聚合输出过 norm+linear+GELU 再入残差）
  S3: 去 FFN + 更新通路
反向线（PCN 原生开关，对照 = qk/h 已知 GOOD）：
  R1: PCN qk/h + FFN
判定：S1 GOOD → FFN 是干扰源（假说成立）；R1 BAD → 双向实锤
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pcn import PCNModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, SEP, H = 32, 31, 24
D, L, HEADS = 64, 12, 4
STEPS = 1000
OUT = Path('results/mechanism')

def gen(B, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    kw = dict(generator=g) if g is not None else {}
    head = torch.randint(0, SEP, (B, H), **kw)
    seq = torch.cat([head, torch.full((B, 1), SEP, dtype=torch.long), head], dim=1)
    m = torch.zeros(B, seq.size(1) - 1); m[:, H:] = 1.0
    return seq[:, :-1].to(DEV), seq[:, 1:].to(DEV), m.to(DEV)

class StdBlock(nn.Module):
    """标准 Pre-LN 块 + sigmoid qk 门，子件可拆"""
    def __init__(self, use_ffn=True, update_path=False):
        super().__init__()
        self.use_ffn, self.update_path = use_ffn, update_path
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        if update_path:
            self.u_ln = nn.LayerNorm(D); self.u_lin = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    def forward(self, x):
        B, N, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, HEADS, D // HEADS)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-1, -2) / math.sqrt(D // HEADS)).mean(1)
        causal = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        g = torch.sigmoid(logits) * causal
        o = torch.einsum('bij,bhjd->bhid', g, v).reshape(B, N, D)
        if self.update_path:
            o = F.gelu(self.u_lin(self.u_ln(o)))
        x = x + self.proj(o)
        if self.use_ffn:
            x = x + self.ffn(self.ln2(x))
        return x

class StdModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(2*H+1, D)
        self.blocks = nn.ModuleList([StdBlock(**kw) for _ in range(L)])
        self.ln_out = nn.LayerNorm(D); self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        N = ids.size(1)
        x = self.emb(ids) + self.pos(torch.arange(N, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))

@torch.no_grad()
def copy_eval(m, pcn=False):
    tots, ns = [], []
    for i in range(3):
        x, y, mm = gen(16, seed=7000+i)
        logits = m(x, topk=16) if pcn else m(x)
        l = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
        cl = l[mm.reshape(-1) > 0.5]
        tots.append(cl.sum().item()); ns.append(cl.numel())
    return sum(tots)/max(sum(ns), 1)

def run_std(kw, seed=0):
    torch.manual_seed(seed)
    m = StdModel(**kw).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(STEPS):
        x, y, _ = gen(32)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return round(copy_eval(m), 3)

def run_pcn_ffn(seed=0):
    torch.manual_seed(seed)
    m = PCNModel(vocab_size=V, d_model=D, n_layers=L, n_heads=HEADS, d_gate=16,
                 dropout=0.0, max_seq_len=2*H+1, init_mode='fixed',
                 gate_source='qk', val_source='wlat_h', use_ffn=True).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(STEPS):
        x, y, _ = gen(32)
        loss = F.cross_entropy(m(x, topk=16).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return round(copy_eval(m, pcn=True), 3)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | 块子件二分 | 3 seed")
    print("=== 加法线（S0 校准必须 BAD>=3）===")
    s0 = [run_std(dict(use_ffn=True), s) for s in range(3)]
    ok0 = all(c > 3.0 for c in s0)
    print(f"  S0 标准块(含FFN): {s0}  {'BAD-confirmed' if ok0 else 'CALIBRATION FAIL'}")
    res = {'S0': s0}
    if ok0:
        for name, kw in (('S1 去FFN', dict(use_ffn=False)),
                         ('S2 更新通路', dict(use_ffn=True, update_path=True)),
                         ('S3 去FFN+更新', dict(use_ffn=False, update_path=True))):
            c = [run_std(kw, s) for s in range(3)]
            okn = sum(1 for x in c if x < 1.0)
            print(f"  {name:<14} {c}  learned {okn}/3")
            res[name] = c
        r1 = [run_pcn_ffn(s) for s in range(3)]
        okr = sum(1 for x in r1 if x < 1.0)
        print(f"  R1 PCN+FFN       {r1}  learned {okr}/3")
        res['R1'] = r1
        json.dump(res, open(OUT/'block_ablation.json','w'), indent=2)
        print(f"saved {OUT}/block_ablation.json")

if __name__ == '__main__':
    main()
