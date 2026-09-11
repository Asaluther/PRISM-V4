#!/usr/bin/env python3
"""MQAR 第四阶段（收尾）：无捷径版——值无放回采样

阶段三后的混淆排查发现：值有放回时「预测众数值」启发式在 n=16 测试集得 0.1263
> gate 臂的 0.115——「门控高于随机」被频率捷径完全解释，非键值绑定证据。
本阶段值改为无放回（每值唯一 → 频率启发式 = 精确随机基线），
重跑关键臂（gate ×2 / tf ×1，D=64、8000步×b64、n=16）确认干净判定。

用法：.venv/Scripts/python.exe mqar_phase4.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from mqar_benchmark import build as build_d64, fwd as fwd_d64, DEV, K_VOCAB, V_VOCAB, SEP, VOCAB, Q

OUT = Path('results/mechanism')
STEPS, BATCH, EVAL_EVERY, D, L, HEADS = 8000, 64, 250, 64, 12, 4


def gen_mqar_unique(B, n, seed=None):
    """与 mqar_benchmark.gen_mqar 同构，唯一差异：值无放回（每行值唯一）"""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    if g is not None:
        keys = torch.stack([torch.randperm(K_VOCAB, generator=g)[:n] for _ in range(B)])
        vals = torch.stack([torch.randperm(V_VOCAB, generator=g)[:n] for _ in range(B)])
        qidx = torch.stack([torch.randperm(n, generator=g)[:Q] for _ in range(B)])
    else:
        keys = torch.stack([torch.randperm(K_VOCAB)[:n] for _ in range(B)])
        vals = torch.stack([torch.randperm(V_VOCAB)[:n] for _ in range(B)])
        qidx = torch.stack([torch.randperm(n)[:Q] for _ in range(B)])
    mem = torch.empty(B, 2 * n, dtype=torch.long)
    mem[:, 0::2] = keys
    mem[:, 1::2] = vals + K_VOCAB
    queries = torch.gather(keys, 1, qidx)
    qvals = torch.gather(vals, 1, qidx)
    seq = torch.cat([mem, torch.full((B, 1), SEP, dtype=torch.long),
                     queries, torch.full((B, 1), SEP, dtype=torch.long)], dim=1)
    x, y = seq[:, :-1], seq[:, 1:].clone()
    qm = torch.zeros_like(y, dtype=torch.bool)
    for i in range(Q):
        y[:, 2 * n + 1 + i] = qvals[:, i] + K_VOCAB
        qm[:, 2 * n + 1 + i] = True
    return x.to(DEV), y.to(DEV), qm.to(DEV)


@torch.no_grad()
def query_acc(model, arm, n):
    model.eval()
    hit, tot = 0, 0
    for i in range(3):
        x, y, qm = gen_mqar_unique(64, n, seed=9000 + i)
        pred = fwd_d64(arm, model, x).argmax(-1)
        mf = qm.reshape(-1)
        hit += (pred.reshape(-1)[mf] == y.reshape(-1)[mf]).sum().item()
        tot += mf.sum().item()
    model.train()
    return hit / tot


def train_one(arm, n, seed, lr):
    torch.manual_seed(seed)
    model = build_d64(arm).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        x, y, _ = gen_mqar_unique(BATCH, n)
        loss = F.cross_entropy(fwd_d64(arm, model, x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            acc = query_acc(model, arm, n)
            curve.append({'step': step, 'acc': round(acc, 3)})
            if acc >= 0.999:
                break
    return curve


def main():
    print(f'===== MQAR 阶段四（值无放回，D={D}，{STEPS}步 × b{BATCH}，n=16）=====')
    print(f'值唯一 → 频率启发式不可用（理论 acc = 1/{V_VOCAB} = {1/V_VOCAB:.4f}）\n')

    results = {'protocol': {'values': '无放回（唯一）', 'd_model': D, 'steps': STEPS,
                            'batch': BATCH, 'n': 16,
                            'note': '阶段一~三的值有放回存在众数值捷径（n=16 天花板 0.1263 > gate 0.115）'}}
    for arm, lr, label, seeds in (('gate', 3e-4, 'PCN 门控@3e-4', [0, 1]),
                                  ('tf', 1e-3, 'TF@1e-3', [0])):
        curves, accs = {}, []
        for s in seeds:
            c = train_one(arm, 16, s, lr)
            curves[f's{s}'] = c
            accs.append(c[-1]['acc'])
            print(f'  {label:<16} n=16 seed{s}: 终末 acc={c[-1]["acc"]:.3f} ({c[-1]["step"]} 步)')
        results[arm] = {'label': label, 'n16_acc': accs,
                        'solved': all(a >= 0.9 for a in accs), 'curves': curves}
        print(f'    -> {"✅ 解出" if results[arm]["solved"] else "❌"}')

    json.dump(results, open(OUT / 'mqar_phase4.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/mqar_phase4.json')


if __name__ == '__main__':
    main()
