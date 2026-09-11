#!/usr/bin/env python3
"""MQAR 第三阶段：容量升级验证（阶段二 gate 双 seed 0.115 爬行 → D=64 是瓶颈假设）

协议：D=128（d_gate=32 等比）、L=12、8000 步 × b64、n=16；gate@3e-4 + TF@1e-3 对照。
gate 解出（≥0.9）→ 追加 n∈{32,48} 阶梯（1 seed）；未解出 → 记录容量边界数据。

用法：.venv/Scripts/python.exe mqar_phase3.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from mqar_benchmark import gen_mqar, DEV, VOCAB, Q
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

OUT = Path('results/mechanism')
D, L, HEADS, DGATE, TOPK = 128, 12, 4, 32, 16
STEPS, BATCH, EVAL_EVERY = 8000, 64, 250


def build(arm):
    if arm == 'tf':
        return TransformerModel(vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=HEADS,
                                ffn_dim=4 * D, dropout=0.0, max_seq_len=256, attn_impl='sdpa')
    return PCNModel(vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=HEADS,
                    d_gate=DGATE, dropout=0.0, max_seq_len=256, init_mode='fixed')


def fwd(arm, m, x):
    return m(x, topk=TOPK) if arm == 'gate' else m(x)


@torch.no_grad()
def query_acc(model, arm, n, batches=3):
    model.eval()
    hit, tot = 0, 0
    for i in range(batches):
        x, y, qm = gen_mqar(64, n, seed=9000 + i)
        pred = fwd(arm, model, x).argmax(-1)
        mf = qm.reshape(-1)
        hit += (pred.reshape(-1)[mf] == y.reshape(-1)[mf]).sum().item()
        tot += mf.sum().item()
    model.train()
    return hit / tot


def train_one(arm, n, seed, lr):
    torch.manual_seed(seed)
    model = build(arm).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        x, y, _ = gen_mqar(BATCH, n)
        loss = F.cross_entropy(fwd(arm, model, x).reshape(-1, VOCAB), y.reshape(-1))
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
    print(f'===== MQAR 阶段三（D={D}/d_gate={DGATE}, {STEPS}步 × b{BATCH}）=====\n')
    results = {'protocol': {'d_model': D, 'd_gate': DGATE, 'topk': TOPK,
                            'steps': STEPS, 'batch': BATCH,
                            'note': '阶段二 D=64 gate 双 seed 0.115 爬行 → 容量瓶颈假设'}}

    for arm, lr, label, seeds in (('tf', 1e-3, 'TF@1e-3', [0]),
                                  ('gate', 3e-4, 'PCN 门控@3e-4', [0, 1])):
        curves, accs = {}, []
        for s in seeds:
            c = train_one(arm, 16, s, lr)
            curves[f's{s}'] = c
            accs.append(c[-1]['acc'])
            print(f'  {label:<16} n=16 seed{s}: 终末 acc={c[-1]["acc"]:.3f} ({c[-1]["step"]} 步)')
        solved = all(a >= 0.9 for a in accs)
        results[arm] = {'label': label, 'n16_acc': accs, 'n16_solved': solved, 'curves': curves}
        print(f'    -> {"✅ 解出" if solved else "❌"}')

    if results['gate']['n16_solved']:
        print('\n----- 门控难度阶梯 -----')
        results['gate_ladder'] = {}
        for n in (32, 48):
            c = train_one('gate', n, 0, 3e-4)
            results['gate_ladder'][f'n{n}'] = {'acc': c[-1]['acc'],
                                               'solved': c[-1]['acc'] >= 0.9,
                                               'curves': {'s0': c}}
            print(f'  PCN 门控 n={n}: 终末 acc={c[-1]["acc"]:.3f} ({c[-1]["step"]} 步)')

    json.dump(results, open(OUT / 'mqar_phase3.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/mqar_phase3.json')


if __name__ == '__main__':
    main()
