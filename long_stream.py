#!/usr/bin/env python3
"""long_stream — 长时程流实验：把流式协议从 8 批/480 步推到 150 批/9000 步

安全框架限速器允许 5000 步/天——既有协议（8 批×60 步=480 步）从未测到它
自己允许预算的量级。本实验用 150 批长流首次测量：

  a. 累积在线增益的形态（净平台 / 衰减 / 转负）
  b. 源域遗忘轨迹（step50 的记忆-遗忘权衡在长程如何演化）
  c. 灾难事件计数（在线 PPL > 冻结基线 2× 的批次数）
  d. ΔW 增长形态（线性 / 饱和）
  e. h2e4 的 50 步 vs 60 步预算净效应（step50 结论的长程检验）

主体（60 步/批 @lr 5e-5）：h2e4_s0（终端默认，+50 步预算臂）、pcn_090m_ckpt
（流式参照王）、tf_ckpt（摧毁参照）。每 10 批探针：源域 PPL（WT 4 序列）、
累积 ΔW（trainable 部分）、用户 held-out。

用法：.venv/Scripts/python.exe long_stream.py   (需 GPU)
输出：results/v7/long_stream.json（逐主体增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import PCN_KW, VOCAB, DEV, build_model
from streaming_090m import adapt_batch, eval_ppl as st_eval
from src.data.wikitext import WikiTextDataset
import numpy as np

N_BATCH = 150
LR = 5e-5
FREEZE_LAYERS = 12
OUT = Path('results/v7')

SUBJECTS = [
    # (标签, 架构, 检查点, 步/批)
    ('h2e4_60', 'hyb', 'results/wt_096m_hyb_h2e-4_s0/best_model.pt', 60),
    ('h2e4_50', 'hyb', 'results/wt_096m_hyb_h2e-4_s0/best_model.pt', 50),
    ('pcn_60', 'pcn', 'results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt', 60),
    ('tf_60', 'tf', 'results/wt_200m_tf_lr5e-4_ckpt/best_model.pt', 60),
]


def load(kind, ckpt):
    m = build_model(kind)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


def trainable_of(model, kind):
    """复用主线冻结语义，返回可训练参数名集合"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    n_freeze = model.layers if kind == 'hyb' else model.layers[:FREEZE_LAYERS]
    for blk in n_freeze:
        for p in blk.parameters():
            p.requires_grad = False
    return {n for n, p in model.named_parameters() if p.requires_grad}


def main():
    print(f'===== 长时程流：{N_BATCH} 批 × (60/50) 步 @lr {LR:g} =====\n')

    # 流数据：主题池扩容到 160 chunks（150 流 + 4 held + 缓冲）
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    tok = GPT2TokenizerFast.from_pretrained(str(Path(__file__).parent / 'tokenizer'))
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, N_BATCH + 10)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(N_BATCH)]
    user_held = chunks_to_xy(pool[N_BATCH:N_BATCH + 4])

    # 源域探针（WT val 前 4 块）
    wt = WikiTextDataset(split='validation', seq_len=256)
    src_probe = {'x': torch.from_numpy(wt.data[:4, :-1].astype('int64')),
                 'y': torch.from_numpy(wt.data[:4, 1:].astype('int64'))}

    results = {'n_batch': N_BATCH, 'lr': LR, 'runs': {}}
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(results, open(OUT / 'long_stream.json', 'w'),
                  indent=2, ensure_ascii=False)

    for tag, kind, ckpt, steps_per_batch in SUBJECTS:
        use_kw = kind in ('pcn', 'hyb')
        kw = PCN_KW if use_kw else None

        @torch.no_grad()
        def ev(m, d):
            if use_kw:
                return st_eval(m, kind, d)
            tot, n = 0.0, 0
            for i in range(d['x'].size(0)):
                logits = m(d['x'][i:i+1].to(DEV))
                l = F.cross_entropy(logits.reshape(-1, VOCAB),
                                    d['y'][i:i+1].to(DEV).reshape(-1),
                                    reduction='sum')
                tot += l.item(); n += d['y'][i:i+1].numel()
            return math.exp(min(tot / n, 20))

        # 冻结基线
        m_frozen = load(kind, ckpt)
        frozen = [ev(m_frozen, b) for b in stream]
        held_frozen = ev(m_frozen, user_held)
        src_frozen = ev(m_frozen, src_probe)
        del m_frozen; torch.cuda.empty_cache()

        m = load(kind, ckpt)
        t_names = trainable_of(m, kind)
        pre = {n: p.detach().clone() for n, p in m.named_parameters()
               if n in t_names}

        traj, probes = [], []
        n_catastrophe = 0
        for i, batch in enumerate(stream):
            ppl = ev(m, batch)
            traj.append(round(ppl, 2))
            if ppl > 2 * frozen[i]:
                n_catastrophe += 1
            if i % 10 == 0 or i == N_BATCH - 1:
                src = ev(m, src_probe)
                held = ev(m, user_held)
                dw = sum((p.detach() - pre[n]).norm().item()
                         for n, p in m.named_parameters() if n in t_names)
                dw0 = sum(pre[n].norm().item() for n in pre)
                probes.append({'batch': i, 'src_ppl': round(src, 2),
                               'src_degr_pct': round((src / src_frozen - 1) * 100, 1),
                               'held_ppl': round(held, 2),
                               'held_gain_pct': round((1 - held / held_frozen) * 100, 1),
                               'rel_dw': round(dw / dw0, 5)})
                print(f'  {tag} 批{i:3d}: 在线 {ppl:8.2f} | 源域 {src:7.2f} '
                      f'(+{probes[-1]["src_degr_pct"]:.0f}%) | held '
                      f'{probes[-1]["held_gain_pct"]:+.1f}% | '
                      f'ΔW {probes[-1]["rel_dw"]*100:.2f}%', flush=True)
            if i < N_BATCH - 1:
                m = adapt_batch(m, kind, batch, steps_per_batch, LR)

        gain_all = (1 - sum(traj) / sum(frozen)) * 100
        gain_tail = (1 - statistics.mean(traj[-30:]) /
                     statistics.mean(frozen[-30:])) * 100
        gain_head = (1 - statistics.mean(traj[:30]) /
                     statistics.mean(frozen[:30])) * 100
        results['runs'][tag] = {
            'steps_per_batch': steps_per_batch,
            'frozen_ppl_head/tail': [round(statistics.mean(frozen[:30]), 1),
                                     round(statistics.mean(frozen[-30:]), 1)],
            'online_gain_all_pct': round(gain_all, 1),
            'online_gain_head30_pct': round(gain_head, 1),
            'online_gain_tail30_pct': round(gain_tail, 1),
            'n_catastrophe': n_catastrophe,
            'final_src_degr_pct': probes[-1]['src_degr_pct'],
            'final_held_gain_pct': probes[-1]['held_gain_pct'],
            'final_rel_dw': probes[-1]['rel_dw'],
            'traj': traj, 'probes': probes}
        print(f'=> {tag}: 全程 {gain_all:+.1f}% | 头30批 {gain_head:+.1f}% | '
              f'尾30批 {gain_tail:+.1f}% | 灾难 {n_catastrophe} 批 | '
              f'源域恶化 {probes[-1]["src_degr_pct"]:+.1f}%\n')
        save()
        del m, pre; torch.cuda.empty_cache()

    print(f'输出: {OUT / "long_stream.json"}')


if __name__ == '__main__':
    main()
