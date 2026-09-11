#!/usr/bin/env python3
"""PRISM V4 — 第 2 步：最小可学习性测试（玩具任务）

决策树（V4_PLAN 修复版）：
  - PCN-fixed 连玩具任务都学不动 → 架构红灯，触发退路
  - 1 层能学、12 层不能 → 深度问题
  - 都能学 → 406 是初始化 bug 伪影 → 进第 3 步重跑

任务设计（vocab=32，排除记忆：每步即时生成新数据）：
  A 周期预测：每序列独立随机 8-token 模式重复 8 次（len=64）。
    pattern 每序列随机 → 位置查表不可解，必须跨位置读前文。unigram 下限 ln32=3.466
  B 复制：24 随机 + SEP + 复制（len=49）。单独统计复制区 loss——
    直接检验误差门控的跨位置聚合是否工作。unigram ln32=3.466

对照矩阵：{PCN-original, PCN-fixed, Transformer} × {1层, 12层} × {任务A, 任务B}
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V = 32
SEP = 31
H = 24  # 复制任务前半长度
STEPS = 1000
OUT = Path('results/diagnosis')


def gen(task, B, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    kw = dict(generator=g) if g is not None else {}
    if task == 'periodic':
        pat = torch.randint(0, V, (B, 8), **kw)
        seq = pat.repeat(1, 8)                       # [B, 64]
        copy_mask = torch.ones(B, seq.size(1) - 1)
    else:  # copy
        head = torch.randint(0, SEP, (B, H), **kw)
        sep = torch.full((B, 1), SEP, dtype=torch.long)
        seq = torch.cat([head, sep, head], dim=1)    # [B, 49]
        m = torch.zeros(B, seq.size(1) - 1)
        m[:, H:] = 1.0                               # 复制区 target
        copy_mask = m
    x, y = seq[:, :-1].to(DEV), seq[:, 1:].to(DEV)
    return x, y, copy_mask.to(DEV)


def build(kind, init_mode, n_layers):
    if kind == 'transformer':
        return TransformerModel(vocab_size=V, d_model=64, n_layers=n_layers,
                                n_heads=4, ffn_dim=256, dropout=0.0,
                                max_seq_len=64).to(DEV)
    return PCNModel(vocab_size=V, d_model=64, n_layers=n_layers, n_heads=4,
                    d_gate=16, dropout=0.0, max_seq_len=64,
                    init_mode=init_mode).to(DEV)


def fwd(model, kind, x):
    if kind == 'pcn':
        return model(x, topk=16)
    return model(x)


@torch.no_grad()
def evaluate(model, kind, task):
    model.eval()
    tots, cops, ns, cns = [], [], [], []
    for i in range(4):
        x, y, m = gen(task, 32, seed=1000 + i)
        logits = fwd(model, kind, x)
        loss_all = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
        mflat = m.reshape(-1)
        tots.append(loss_all.sum().item()); ns.append(loss_all.numel())
        cl = loss_all[mflat > 0.5]
        cops.append(cl.sum().item()); cns.append(cl.numel())
    model.train()
    return sum(tots) / sum(ns), sum(cops) / max(sum(cns), 1)


@torch.no_grad()
def gate_selectivity(model, task='periodic'):
    """训练后门控 g 的 std —— 选择性是否长出来（死锁时 ≈0）"""
    model.eval()
    x, _, _ = gen(task, 4, seed=7)
    captured = {}

    def pre_hook(idx):
        def hook(module, args):
            captured[idx] = args[0].detach()
        return hook

    hs = [b.register_forward_pre_hook(pre_hook(i)) for i, b in enumerate(model.layers)]
    with torch.no_grad():
        model(x, topk=16)
    for h in hs:
        h.remove()
    # 复算中层门控
    i = min(5, len(model.layers) - 1)
    pc = model.layers[i].pcn_layer
    x_i = captured[i]
    u = pc.W_bu(pc.norm_bu(x_i))
    p = pc.W_td(captured[i - 1]) if i > 0 else torch.zeros_like(u)
    e = (u - p).clamp(-10, 10)
    e_comp = pc.W_compress(e)
    if pc.gate_ln:
        e_comp = F.layer_norm(e_comp, (e_comp.size(-1),))
    e_i, e_j = e_comp.unsqueeze(2), e_comp.unsqueeze(1)
    inter = torch.cat([e_i * e_j, e_i + e_j], dim=-1)
    g = torch.sigmoid(pc.w_g(inter).squeeze(-1))
    model.train()
    return g.std().item()


def run_one(kind, init_mode, n_layers, task):
    torch.manual_seed(0)
    model = build(kind, init_mode, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        x, y, _ = gen(task, 32)
        logits = fwd(model, kind, x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0:
            tot, cop = evaluate(model, kind, task)
            curve.append({'step': step, 'loss': tot, 'copy_loss': cop})
    tot, cop = curve[-1]['loss'], curve[-1]['copy_loss']
    gs = gate_selectivity(model) if kind == 'pcn' else None
    name = f"{kind}{'-' + init_mode if kind == 'pcn' else ''} L{n_layers} [{task}]"
    print(f"  {name:<38} loss={tot:.3f}  copy={cop:.3f}  gate_std={gs if gs is None else round(gs, 4)}")
    return {'name': name, 'curve': curve, 'gate_std': gs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    unigram = math.log(V)
    print(f"Device: {DEV} | unigram 下限 = {unigram:.3f} | steps={STEPS}\n")
    results = []
    for task in ('periodic', 'copy'):
        for kind, mode in (('transformer', None), ('pcn', 'original'), ('pcn', 'fixed')):
            for L in (1, 12):
                results.append(run_one(kind, mode, L, task))
        print()
    with open(OUT / 'step2_toy.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"报告: {OUT / 'step2_toy.json'}")


if __name__ == '__main__':
    main()
