#!/usr/bin/env python3
"""PRISM R2 — 复制任务机理解剖

现象：因果 attention 1000 步学不会精确复制（copy loss 恒 unigram），误差门控会（0.17）。
要回答：机制根源是「softmax 竞争归一化」还是别的？

2a 动力学解剖：复制训练中每 100 步记录 tf_softmax 的
  - 指向性质量：复制区位置 i 对正确信息源 j=i-H-1 的 attention 权重
  - attention 熵（因果范围）
2b attention 干预：softmax | sigmoid 门控式 | 温度 0.5/2.0
2c 门控反向干预：PCN 独立 sigmoid | PCN softmax 化

若 tf_sigmoid 学会复制且 pcn_softmax 学不会 → 机制锁定为竞争归一化 vs 独立门。
用法：CUDA_VISIBLE_DEVICES=-1 python copy_mechanism.py   # CPU 与 R1 并行
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pcn import PCNModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, SEP, H = 32, 31, 24
D, L, HEADS = 64, 12, 4
STEPS, EVAL_EVERY = 1000, 100
OUT = Path('results/mechanism')


def gen(B, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    kw = dict(generator=g) if g is not None else {}
    head = torch.randint(0, SEP, (B, H), **kw)
    seq = torch.cat([head, torch.full((B, 1), SEP, dtype=torch.long), head], dim=1)
    m = torch.zeros(B, seq.size(1) - 1)
    m[:, H:] = 1.0
    return seq[:, :-1].to(DEV), seq[:, 1:].to(DEV), m.to(DEV)


# ============ 变体 Transformer（2b）============

class VariantAttnBlock(nn.Module):
    """Pre-LN 块，聚合函数可变；保存 last_w 供追踪"""
    def __init__(self, mode='softmax', temp=1.0):
        super().__init__()
        self.mode, self.temp = mode, temp
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.last_w = None

    def forward(self, x):
        B, N, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, HEADS, D // HEADS)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ k.transpose(-1, -2) / math.sqrt(D // HEADS)
        causal = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        if self.mode == 'softmax':
            w = torch.softmax(logits.masked_fill(~causal, float('-inf')) / self.temp, dim=-1)
        else:  # sigmoid：独立门控，无归一化
            w = torch.sigmoid(logits / self.temp) * causal
        self.last_w = w.detach()
        o = (w @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(o)
        x = x + self.ffn(self.ln2(x))
        return x


class VariantTransformer(nn.Module):
    def __init__(self, mode='softmax', temp=1.0):
        super().__init__()
        self.token_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(2 * H + 1, D)
        self.blocks = nn.ModuleList([VariantAttnBlock(mode, temp) for _ in range(L)])
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.token_emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        N = ids.size(1)
        x = self.token_emb(ids) + self.pos_emb(torch.arange(N, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_out(x))


# ============ 追踪指标（2a）============

@torch.no_grad()
def pointability_tf(model):
    """复制区对正确偏移位置的 attention 质量 + 熵（中层块，所有头平均）"""
    blk = model.blocks[L // 2]
    w = blk.last_w  # [B, Hh, N, N]
    idx = torch.arange(w.size(-1), device=w.device)
    j = (idx - H - 1).clamp(min=0)
    rows = idx >= H + 1
    tgt = w[:, :, rows, :][:, :, torch.arange(rows.sum().item(), device=w.device), j[rows]]
    p = w[:, :, rows, :]
    ent = -(p.clamp_min(1e-10).log() * p).sum(-1).mean().item()
    return tgt.mean().item(), ent


@torch.no_grad()
def copy_loss_eval(model, kind):
    tots, ns = [], []
    for i in range(3):
        x, y, m = gen(16, seed=7000 + i)
        logits = model(x) if kind.startswith('tf') else model(x, topk=16)
        l = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
        mf = m.reshape(-1) > 0.5
        cl = l[mf]
        tots.append(cl.sum().item()); ns.append(cl.numel())
    return sum(tots) / max(sum(ns), 1)


def train_variant(name, make_model, kind, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    model = make_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        x, y, _ = gen(32)
        logits = model(x) if kind.startswith('tf') else model(x, topk=16)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            cl = copy_loss_eval(model, kind)
            rec = {'step': step, 'copy_loss': round(cl, 3)}
            if kind == 'tf_softmax':
                with torch.no_grad():
                    xe, ye, _ = gen(16, seed=7)
                    model(xe)
                pt, ent = pointability_tf(model)
                rec['pointability'], rec['entropy'] = round(pt, 4), round(ent, 3)
            curve.append(rec)
    final = curve[-1]['copy_loss']
    print(f"  {name:<22} 最终 copy={final:.3f}  学会={'✅' if final < 1.0 else '❌'}")
    return {'name': name, 'curve': curve}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ns = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | 复制任务 H={H} | {L}L-d{D} | {STEPS} 步 | seed={ns.seed}\n")
    print("===== 2a/2b/2c 变体矩阵 =====")
    results = []

    results.append(train_variant('tf_softmax(原版)', lambda: VariantTransformer('softmax'), 'tf_softmax', seed=ns.seed))
    results.append(train_variant('tf_sigmoid(门控式)', lambda: VariantTransformer('sigmoid'), 'tf_sigmoid', seed=ns.seed))
    results.append(train_variant('tf_sm_t0.5', lambda: VariantTransformer('softmax', 0.5), 'tf_sm_t0.5', seed=ns.seed))
    results.append(train_variant('tf_sm_t2.0', lambda: VariantTransformer('softmax', 2.0), 'tf_sm_t2.0', seed=ns.seed))

    def make_pcn(gs):
        return PCNModel(vocab_size=V, d_model=D, n_layers=L, n_heads=HEADS,
                        d_gate=16, dropout=0.0, max_seq_len=2 * H + 1,
                        init_mode='fixed', gate_softmax=gs)

    results.append(train_variant('pcn_sigmoid(原版)', lambda: make_pcn(False), 'pcn', seed=ns.seed))
    results.append(train_variant('pcn_softmax(2c)', lambda: make_pcn(True), 'pcn_softmax', seed=ns.seed))

    with open(OUT / f'copy_mechanism_s{ns.seed}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n===== 判读 =====")
    fin = {r['name']: r['curve'][-1]['copy_loss'] for r in results}
    sig_tf, sm_ng = fin['tf_sigmoid(门控式)'], fin['pcn_softmax(2c)']
    if sig_tf < 1.0 and sm_ng > 2.0:
        print("  ✅ 机制锁定：竞争归一化(softmax)学不会、独立门(sigmoid)会——与架构无关")
    elif sig_tf > 2.0 and sm_ng > 2.0:
        print("  ⚠️ sigmoid attention 也没学会：差异不止归一化方式，误差流本身有贡献")
    elif sig_tf < 1.0 and sm_ng < 1.0:
        print("  ⚠️ softmax 门控也学会了：机制不在归一化，看指向性曲线")
    else:
        print("  ❓ 混合，看 json 曲线")
    print(f"\n输出: {OUT}/copy_mechanism.json")


if __name__ == '__main__':
    main()
