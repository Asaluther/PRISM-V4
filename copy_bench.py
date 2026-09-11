#!/usr/bin/env python3
"""PRISM V4 — V4.x#3：复制任务优势正式化

把 toy_test 的初步信号（PCN 1000 步学会复制、同规模 Transformer 未学会）
升格为正式证据：多长度 × 多 seed × 步数曲线。

协议：
  - 任务：前 H 个随机 token + SEP + 复制（len = 2H+1），vocab=32
  - 长度 H ∈ {16, 32, 64}；seed ∈ 1..5（模型初始化 + 数据顺序）
  - 模型：PCN-fixed 12L vs Transformer 12L（同 toy_test 规格：d=64）
  - 1000 步，每 100 步评 held-out copy-region loss
  - 「学会」判据：copy-region loss < 1.0（unigram 下限 ln32=3.47）

输出：results/analysis_v2/copy_bench.json + 控制台汇总
用法：CUDA_VISIBLE_DEVICES=-1 python copy_bench.py   # CPU 并行跑（GPU 留给 #1）
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
STEPS = 1000
EVAL_EVERY = 100
LEARN_THR = 1.0
OUT = Path('results/analysis_v2')


def gen(H, B, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    kw = dict(generator=g) if g is not None else {}
    head = torch.randint(0, SEP, (B, H), **kw)
    seq = torch.cat([head, torch.full((B, 1), SEP, dtype=torch.long), head], dim=1)
    m = torch.zeros(B, seq.size(1) - 1)
    m[:, H:] = 1.0
    x, y = seq[:, :-1].to(DEV), seq[:, 1:].to(DEV)
    return x, y, m.to(DEV)


def build(kind, max_len):
    if kind == 'transformer':
        return TransformerModel(vocab_size=V, d_model=64, n_layers=12, n_heads=4,
                                ffn_dim=256, dropout=0.0, max_seq_len=max_len).to(DEV)
    return PCNModel(vocab_size=V, d_model=64, n_layers=12, n_heads=4,
                    d_gate=16, dropout=0.0, max_seq_len=max_len,
                    init_mode='fixed').to(DEV)


@torch.no_grad()
def eval_copy(model, kind, H):
    model.eval()
    tots, ns = [], []
    for i in range(3):
        x, y, m = gen(H, 16, seed=9000 + i)
        logits = model(x, topk=16) if kind == 'pcn' else model(x)
        l = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
        mf = m.reshape(-1) > 0.5
        cl = l[mf]
        tots.append(cl.sum().item()); ns.append(cl.numel())
    model.train()
    return sum(tots) / max(sum(ns), 1)


def run(kind, H, seed, lr=3e-4):
    torch.manual_seed(seed)
    model = build(kind, 2 * H + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve, steps_to_learn = [], None
    for step in range(1, STEPS + 1):
        x, y, _ = gen(H, 32)
        logits = model(x, topk=16) if kind == 'pcn' else model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            c = eval_copy(model, kind, H)
            curve.append(round(c, 3))
            if steps_to_learn is None and c < LEARN_THR:
                steps_to_learn = step
    return {'final_copy_loss': curve[-1], 'steps_to_learn': steps_to_learn,
            'curve': curve}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--only', type=str, default=None,
                    help='只跑指定模型（transformer/pcn），用于 lr 复检')
    ns = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | lr={ns.lr} | 学会判据: copy loss < {LEARN_THR} | {STEPS} 步上限\n")
    results = []
    kinds = (ns.only,) if ns.only else ('pcn', 'transformer')
    for H in (16, 32, 64):
        for kind in kinds:
            for seed in range(1, 6):
                r = run(kind, H, seed, lr=ns.lr)
                r.update({'model': kind, 'H': H, 'seed': seed})
                results.append(r)
                stl = r['steps_to_learn'] if r['steps_to_learn'] else f">{STEPS}"
                print(f"  H={H:<3} {kind:<12} seed{seed}  copy={r['final_copy_loss']:.3f}  学会于 {stl} 步")

    # 汇总
    print("\n===== 汇总（5 seed）=====")
    summary = {}
    for H in (16, 32, 64):
        for kind in kinds:
            rs = [r for r in results if r['H'] == H and r['model'] == kind]
            if not rs:
                continue
            learned = [r['steps_to_learn'] for r in rs if r['steps_to_learn']]
            summary[f'{kind}_H{H}'] = {
                'learned_rate': f"{len(learned)}/5",
                'median_steps_to_learn': sorted(learned)[len(learned)//2] if learned else None,
                'mean_final_copy_loss': round(sum(r['final_copy_loss'] for r in rs) / len(rs), 3),
            }
            print(f"  H={H:<3} {kind:<12} 学会率 {summary[f'{kind}_H{H}']['learned_rate']}  "
                  f"中位步数 {summary[f'{kind}_H{H}']['median_steps_to_learn']}  "
                  f"平均终值 {summary[f'{kind}_H{H}']['mean_final_copy_loss']}")

    tag = f"_lr{ns.lr:g}" + (f"_{ns.only}" if ns.only else "")
    with open(OUT / f'copy_bench{tag}.json', 'w') as f:
        json.dump({'results': results, 'summary': summary,
                   'protocol': {'steps': STEPS, 'learn_thr': LEARN_THR,
                                'unigram': math.log(V), 'd_model': 64, 'n_layers': 12}}, f, indent=2)
    print(f"\n输出: {OUT}/copy_bench.json")


if __name__ == '__main__':
    main()
