#!/usr/bin/env python3
"""PRISM R2 补充 — 门来源 × 值通路 2×2 分解

已有结果（copy_mechanism）：
  qk门 + V值（tf_sigmoid）      → 学不会 ❌
  feat门 + W_lat值（pcn 原版）  → 学会 ✅
缺的两格（本脚本，近似实现——Transformer 块语境下的等价物）：
  qk门 + W_lat值：门来自 q·k，值来自独立的 W_lat(h) 投影（非 QKV 绑定）
  feat门 + V值：  门来自压缩特征交互，值走标准 V 投影
判定：必要条件在「门来源」还是「值通路」（还是必须两者同时）。
用法：CUDA_VISIBLE_DEVICES=-1 python gate_source_test.py
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class HybridBlock(nn.Module):
    """门来源（qk/feat）× 值通路（v/wlat）的 2×2 变体块"""
    def __init__(self, gate_src, val_src):
        super().__init__()
        self.gate_src, self.val_src = gate_src, val_src
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.wlat = nn.Linear(D, D)      # W_lat 等价物：独立值投影
        self.dg_in = nn.Linear(D, DG)    # feat 门：压缩投影
        self.w_g = nn.Linear(2 * DG, 1)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        B, N, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, HEADS, D // HEADS)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        causal = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))

        if self.gate_src == 'qk':
            logits = (q @ k.transpose(-1, -2) / math.sqrt(D // HEADS)).mean(1)  # [B,N,N]
            g = torch.sigmoid(logits)
        else:  # feat：压缩特征交互（PCN 同式）
            e = self.dg_in(h)                                        # [B,N,DG]
            e_i, e_j = e.unsqueeze(2), e.unsqueeze(1)
            g = torch.sigmoid(self.w_g(torch.cat([e_i * e_j, e_i + e_j], -1)).squeeze(-1))
        g = g * causal

        if self.val_src == 'v':
            # 全头值（g 广播到所有头，拼回 D 维——夜间版本的 v.mean(1) 丢维 bug 已修）
            o = torch.einsum('bij,bhjd->bhid', g, v).reshape(B, N, D)
            x = x + self.proj(o)
        else:  # wlat：独立投影值
            val = self.wlat(h)
            o = torch.einsum('bij,bjd->bid', g, val)
            x = x + self.proj(o)
        x = x + self.ffn(self.ln2(x))
        return x


class HybridModel(nn.Module):
    def __init__(self, gate_src, val_src):
        super().__init__()
        self.token_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(2 * H + 1, D)
        self.blocks = nn.ModuleList([HybridBlock(gate_src, val_src) for _ in range(L)])
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.token_emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            nn.init.xavier_uniform_(b.w_g.weight)

    def forward(self, ids):
        N = ids.size(1)
        x = self.token_emb(ids) + self.pos_emb(torch.arange(N, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))


@torch.no_grad()
def copy_loss_eval(model):
    tots, ns = [], []
    for i in range(3):
        x, y, m = gen(16, seed=7000 + i)
        l = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction='none')
        cl = l[m.reshape(-1) > 0.5]
        tots.append(cl.sum().item()); ns.append(cl.numel())
    return sum(tots) / max(sum(ns), 1)


def run(name, gate_src, val_src, seed=0):
    torch.manual_seed(seed)
    model = HybridModel(gate_src, val_src).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(1, STEPS + 1):
        x, y, _ = gen(32)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    cl = copy_loss_eval(model)
    print(f"  {name:<28} 最终 copy={cl:.3f}  学会={'✅' if cl < 1.0 else '❌'}")
    return {'name': name, 'copy_loss': round(cl, 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | 2×2 门来源×值通路 | {L}L-d{D} | {STEPS} 步\n")
    results = [
        run('qk门+V值（已知❌对照）', 'qk', 'v'),
        run('qk门+W_lat值', 'qk', 'wlat'),
        run('feat门+V值', 'feat', 'v'),
        run('feat门+W_lat值（已知✅对照）', 'feat', 'wlat'),
    ]
    with open(OUT / 'gate_source_2x2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n输出: {OUT}/gate_source_2x2.json")


if __name__ == '__main__':
    main()
