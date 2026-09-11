#!/usr/bin/env python3
"""遗忘基准前置校准：为各架构×各臂选「适应增益最大」的 (lr, steps)

问题：直接沿用 demo/skeleton 的 lr 5e-4×300 步在域内 base 上是纯破坏
（held_A 19.5→64）。demo 的增益来自跨域场景。按项目公平纪律（教训 8：
各侧在自己最优超参上对比），先为 PCN/TF 各自选出适应增益最大的 (lr, steps)，
正式遗忘基准（calibrated regime）用各自最优设置跑，协议其余部分对称。

  arm1 校准目标：FT(A=奇幻池) 后 held_A PPL 改善最大（域内适应）
  arm2 校准目标：FT(WT) 后 held_WT PPL 改善最大（跨域适应）

用法：.venv/Scripts/python.exe forgetting_calibration.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from forgetting_benchmark import (build_topic_pool, chunks_to_xy, load_model,
                                  eval_ppl, forward_logits, DEV, KW_A, KW_B,
                                  N_TRAIN, N_HELD, VOCAB)
from src.data.wikitext import WikiTextDataset

SEED = 1
LRS = [5e-5, 1e-4, 2.5e-4, 5e-4]
STEPSS = [60, 150, 300]
OUT = Path('results/mechanism')


def finetune(model, kind, train_data, steps, lr):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = forward_logits(model, kind, x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool_a = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    train_a, held_a = chunks_to_xy(pool_a[:N_TRAIN]), chunks_to_xy(pool_a[-N_HELD:])
    wt = WikiTextDataset(split='train', seq_len=256)
    import numpy as np
    train_wt = chunks_to_xy(wt.data[5000:5000+N_TRAIN].astype(np.int64))
    held_wt = chunks_to_xy(wt.data[6000:6000+N_HELD].astype(np.int64))

    targets = {'arm1': (train_a, held_a), 'arm2': (train_wt, held_wt)}
    results = {}
    for arm, (train, held) in targets.items():
        results[arm] = {}
        for kind in ('pcn', 'tf'):
            p0 = eval_ppl(load_model(kind, SEED), kind, held)
            print(f'{arm} {kind}: base PPL = {p0:.2f}')
            results[arm][kind] = {'base_ppl': round(p0, 2), 'grid': {}}
            for lr in LRS:
                for steps in STEPSS:
                    m = finetune(load_model(kind, SEED), kind, train, steps, lr)
                    p1 = eval_ppl(m, kind, held)
                    imp = (1 - p1 / p0) * 100
                    results[arm][kind]['grid'][f'lr{lr}_s{steps}'] = {
                        'ppl': round(p1, 2), 'improve_pct': round(imp, 1)}
                    print(f'  lr={lr:<7} steps={steps:<3} -> {p1:9.2f}  ({imp:+6.1f}%)')
            best = max(results[arm][kind]['grid'].items(),
                       key=lambda kv: kv[1]['improve_pct'])
            results[arm][kind]['best'] = {'config': best[0], **best[1]}
            print(f'  ★ {arm} {kind} 最优: {best[0]} ({best[1]["improve_pct"]:+.1f}%)\n')

    json.dump(results, open(OUT / 'forgetting_calibration.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'输出: {OUT}/forgetting_calibration.json')


if __name__ == '__main__':
    main()
