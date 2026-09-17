#!/usr/bin/env python3
"""seed_replication — 3-seed × 3-架构协议矩阵（论文级 seed 稳健性复现）

预注册判据（SEED_REPLICATION 计划）：
  1. 预训练：每 seed HYB PPL 比 TF 低 >20%，且低于 PCN
  2. 单发冷启动：每 seed HYB ≥+30%
  3. 流式：每 seed HYB @5e-5 ≥+15%（@1e-4 允许负值，已知阈值移动）

复用：coldstart_090m 的数据构造/评估/微调 + streaming_090m 的流式机制；
不改动已归档的单 seed 脚本。缺检查点的 (架构, seed) 格自动跳过并报告。

用法：.venv/Scripts/python.exe seed_replication.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from coldstart_090m import (PCN_KW, VOCAB, DEV, FREEZE_LAYERS, OUT,
                            make_data, build_model,
                            eval_ppl as cs_eval, finetune as cs_finetune)
from streaming_090m import adapt_batch, eval_ppl as st_eval, STEPS_PER_BATCH
from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
from src.data.tinystories import TinyStoriesDataset
from src.data.wikitext import WikiTextDataset

SEEDS = [0, 1, 2]
FT_LRS = [1e-4, 5e-5, 2.5e-4]

CKPTS = {
    'hyb': ['results/wt_094m_hyb_s0/best_model.pt',
            'results/wt_094m_hyb_s1/best_model.pt',
            'results/wt_094m_hyb_s2/best_model.pt'],
    'tf':  ['results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
            'results/wt_200m_tf_lr5e-4_s1/best_model.pt',
            'results/wt_200m_tf_lr5e-4_s2/best_model.pt'],
    'pcn': ['results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt',
            'results/wt_090m_pcn_lr1e-4_s1/best_model.pt',
            'results/wt_090m_pcn_lr1e-4_s2/best_model.pt'],
}
PRETRAIN_JSON = {
    'hyb': ['results/wt_094m_hyb_s0/results.json',
            'results/wt_094m_hyb_s1/results.json',
            'results/wt_094m_hyb_s2/results.json'],
    'tf':  ['results/wt_200m_tf_lr5e-4_ckpt/results.json',
            'results/wt_200m_tf_lr5e-4_s1/results.json',
            'results/wt_200m_tf_lr5e-4_s2/results.json'],
    'pcn': ['results/wt_090m_pcn_lr1e-4_ckpt/results.json',
            'results/wt_090m_pcn_lr1e-4_s1/results.json',
            'results/wt_090m_pcn_lr1e-4_s2/results.json'],
}


def load_model(kind, ckpt):
    m = build_model(kind)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


def run_stream(kind, ckpt, stream, gen, user_held, wt_held, lr):
    """streaming_090m.run_stream 的显式检查点版"""
    model = load_model(kind, ckpt)
    traj = []
    pre = {n: p.detach().clone() for n, p in model.named_parameters()
           if n.startswith(('layers.', 'pcn_blocks.'))}
    for i, batch in enumerate(stream):
        traj.append(round(st_eval(model, kind, batch), 2))
        if i < len(stream) - 1:
            model = adapt_batch(model, kind, batch, STEPS_PER_BATCH, lr)
    deltas = [(p.detach() - pre[n]).norm().item() for n, p in model.named_parameters()
              if n in pre and (n.startswith('pcn_blocks.')
                               or int(n.split('.')[1]) >= FREEZE_LAYERS)]
    held_adapted = st_eval(model, kind, user_held)
    return traj, held_adapted, round(sum(deltas) / len(deltas), 4)


def mean_std(xs):
    return (round(statistics.mean(xs), 2),
            round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0)


def main():
    print('===== 3-seed × 3-架构复现矩阵 =====\n')

    # ---- 预训练 PPL ----
    pretrain = {}
    print('--- 预训练 PPL ---')
    for arch in CKPTS:
        vals = []
        for s, p in zip(SEEDS, PRETRAIN_JSON[arch]):
            if Path(p).exists():
                vals.append(json.load(open(p))['best_ppl'])
            else:
                print(f'  [缺] {p}')
        if vals:
            m, sd = mean_std(vals)
            pretrain[arch] = {'per_seed': vals, 'mean': m, 'std': sd}
            print(f'  {arch:4s}: ' + ' / '.join(f'{v:.2f}' for v in vals)
                  + f'  → {m} ± {sd}')

    # ---- 单发冷启动 ----
    print('\n--- 单发冷启动（3 用户/臂/seed）---')
    users, tests = make_data()
    coldstart = {}
    for arch in CKPTS:
        per_seed = []
        for s, ckpt in zip(SEEDS, CKPTS[arch]):
            if not Path(ckpt).exists():
                print(f'  [缺] {ckpt}')
                per_seed.append(None)
                continue
            use_kw = arch != 'tf'
            imps = []
            for u in range(len(users)):
                m = load_model(arch, ckpt)
                pb = cs_eval(m, tests[u], use_kw)
                m_ft = cs_finetune(m, users[u], use_kw)
                pa = cs_eval(m_ft, tests[u], use_kw)
                imps.append((1 - pa / pb) * 100)
            per_seed.append(round(statistics.mean(imps), 1))
            print(f'  {arch} s{s}: {per_seed[-1]:+.1f}%')
            torch.cuda.empty_cache()
        vals = [v for v in per_seed if v is not None]
        if vals:
            m, sd = mean_std(vals)
            coldstart[arch] = {'per_seed': per_seed, 'mean': m, 'std': sd}

    # ---- 流式 ----
    print('\n--- 流式（prequential，3 lr）---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])
    val_ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    gen = chunks_to_xy(torch.stack(val_ds.data[10:14]).numpy())
    wt = WikiTextDataset(split='train', seq_len=256)
    wt_held = chunks_to_xy(wt.data[6000:6004].astype(np.int64))

    streaming = {}
    for arch in CKPTS:
        for s, ckpt in zip(SEEDS, CKPTS[arch]):
            if not Path(ckpt).exists():
                continue
            m_frozen = load_model(arch, ckpt)
            frozen = [st_eval(m_frozen, arch, b) for b in stream]
            held_frozen = st_eval(m_frozen, arch, user_held)
            del m_frozen; torch.cuda.empty_cache()
            for lr in FT_LRS:
                traj, held_a, dw = run_stream(arch, ckpt, stream, gen,
                                              user_held, wt_held, lr)
                gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
                held_gain = (1 - held_a / held_frozen) * 100
                streaming[f'{arch}_s{s}_lr{lr:g}'] = {
                    'online_gain_pct': round(gain, 1),
                    'heldout_gain_pct': round(held_gain, 1),
                    'cum_dw': dw}
                print(f'  {arch} s{s} lr={lr:g}: 在线 {gain:+.1f}%  '
                      f'held-out {held_gain:+.1f}%  ΔW {dw}')
                torch.cuda.empty_cache()

    json.dump({'pretrain': pretrain, 'coldstart': coldstart,
               'streaming': streaming},
              open(OUT / 'seed_replication.json', 'w'), indent=2, ensure_ascii=False)

    # ---- 预注册判据判定 ----
    print('\n===== 预注册判据 =====')
    ok = []
    if 'hyb' in pretrain and 'tf' in pretrain:
        gaps = [(1 - h / t) * 100 for h, t in
                zip(pretrain['hyb']['per_seed'], pretrain['tf']['per_seed'])
                if h is not None and t is not None]
        if gaps:
            hit = all(g > 20 for g in gaps)
            ok.append(hit)
            print(f'  ① 预训练 HYB<TF 差距: ' +
                  ' / '.join(f'{g:.1f}%' for g in gaps) +
                  f' → {"✅ 全 seed >20%" if hit else "❌"}')
    if 'hyb' in coldstart:
        vals = [v for v in coldstart['hyb']['per_seed'] if v is not None]
        if vals:
            hit = all(v >= 30 for v in vals)
            ok.append(hit)
            print(f'  ② 单发 HYB ≥+30%: {vals} → {"✅" if hit else "❌"}')
    hyb55 = [streaming[k]['online_gain_pct'] for k in streaming
             if k.startswith('hyb') and k.endswith('lr5e-05')]
    if hyb55:
        hit = all(v >= 15 for v in hyb55)
        ok.append(hit)
        print(f'  ③ 流式 HYB@5e-5 ≥+15%: {hyb55} → {"✅" if hit else "❌"}')

    print(f'\n输出: {OUT / "seed_replication.json"}')
    print(f'总判定: {sum(ok)}/{len(ok)} 判据命中')


if __name__ == '__main__':
    main()
