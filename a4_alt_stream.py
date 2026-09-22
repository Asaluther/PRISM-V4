#!/usr/bin/env python3
"""a4_alt_stream — 多主题交替流：遗忘-再学习动态（A4，终端多域共存）

长程流（long_stream.py）用单一主题池。本实验用两个不相交主题池
（A=KW_A 关键词、B=KW_B 关键词）构造**交替流**：ABABAB...各 75 批，
测多域共存能力。

预注册判读：
  a. 主题切换后的「切换成本」：批 t（主题 A）紧跟批 t-1（主题 B）的在线
     PPL 是否比同主题连续批更高？
  b. 累积 A 域知识是否随 B 域学习而遗忘（A 探针轨迹）？
  c. 混合架构 vs 纯 PCN 的多域共存差异

主体：h2e4_s0（终端默认）+ pcn_090m_ckpt（参照）@lr 5e-5, 60 步/批。

用法：.venv/Scripts/python.exe a4_alt_stream.py   (需 GPU)
输出：results/v7/a4_alt_stream.json
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

N_PER_TOPIC = 75   # 每主题 75 批 → 总 150 批
LR = 5e-5
STEPS = 60
OUT = Path('results/v7')

SUBJECTS = [
    ('h2e4', 'hyb', 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'),
    ('pcn', 'pcn', 'results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt'),
]


def load(kind, ckpt):
    m = build_model(kind)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def ev(m, kind, d):
    return st_eval(m, kind, d)


def main():
    print(f'===== A4：多主题交替流（{N_PER_TOPIC}×2 = {N_PER_TOPIC*2} 批, '
          f'@lr {LR:g}, {STEPS} 步/批）=====')

    # 构造两个不相交主题池
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    tok = GPT2TokenizerFast.from_pretrained(str(Path(__file__).parent / 'tokenizer'))
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool_a = build_topic_pool(ds, tok, KW_A, KW_B, N_PER_TOPIC + 10)
    pool_b = build_topic_pool(ds, tok, KW_B, KW_A, N_PER_TOPIC + 10)

    # 交替流：A[0] B[0] A[1] B[1] ...
    stream = []
    topics = []
    for i in range(N_PER_TOPIC):
        stream.append(chunks_to_xy(pool_a[i:i+1]))
        topics.append('A')
        stream.append(chunks_to_xy(pool_b[i:i+1]))
        topics.append('B')

    # 探针：各域 held-out + 源域
    probe_a = chunks_to_xy(pool_a[N_PER_TOPIC:N_PER_TOPIC+4])
    probe_b = chunks_to_xy(pool_b[N_PER_TOPIC:N_PER_TOPIC+4])
    wt = WikiTextDataset(split='validation', seq_len=256)
    src = {'x': torch.from_numpy(wt.data[:4, :-1].astype('int64')),
           'y': torch.from_numpy(wt.data[:4, 1:].astype('int64'))}

    results = {'n_per_topic': N_PER_TOPIC, 'lr': LR, 'steps': STEPS,
               'stream_pattern': 'ABAB', 'subjects': {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for tag, kind, ckpt in SUBJECTS:
        print(f'\n--- {tag} ---', flush=True)

        # 冻结基线
        m_frozen = load(kind, ckpt)
        frozen = [ev(m_frozen, kind, b) for b in stream]
        pa0 = ev(m_frozen, kind, probe_a)
        pb0 = ev(m_frozen, kind, probe_b)
        src0 = ev(m_frozen, kind, src)
        del m_frozen; torch.cuda.empty_cache()

        m = load(kind, ckpt)
        traj = []
        probes = []
        for i, batch in enumerate(stream):
            ppl = ev(m, kind, batch)
            traj.append(round(ppl, 2))
            if i % 10 == 0 or i == len(stream) - 1:
                pa = ev(m, kind, probe_a)
                pb = ev(m, kind, probe_b)
                s = ev(m, kind, src)
                probes.append({'batch': i, 'topic': topics[i],
                               'probe_a': round(pa, 1),
                               'probe_a_gain': round((1 - pa / pa0) * 100, 1),
                               'probe_b': round(pb, 1),
                               'probe_b_gain': round((1 - pb / pb0) * 100, 1),
                               'src': round(s, 1),
                               'src_degr': round((s / src0 - 1) * 100, 1)})
                print(f'  批{i:3d}({topics[i]}): A探针 {pa:7.1f} '
                      f'({probes[-1]["probe_a_gain"]:+.1f}%) | '
                      f'B探针 {pb:7.1f} ({probes[-1]["probe_b_gain"]:+.1f}%) | '
                      f'源域 {s:7.1f} (+{probes[-1]["src_degr"]:.0f}%)',
                      flush=True)
            if i < len(stream) - 1:
                m = adapt_batch(m, kind, batch, STEPS, LR)

        # 切换成本分析：主题 A 批紧跟 B 批的 PPL vs A 批紧跟 A 批的 PPL
        a_after_b = [traj[i] for i in range(len(traj))
                     if topics[i] == 'A' and i > 0 and topics[i-1] == 'B']
        a_after_a = [traj[i] for i in range(len(traj))
                     if topics[i] == 'A' and i > 0 and topics[i-1] == 'A']
        b_after_a = [traj[i] for i in range(len(traj))
                     if topics[i] == 'B' and i > 0 and topics[i-1] == 'A']
        b_after_b = [traj[i] for i in range(len(traj))
                     if topics[i] == 'B' and i > 0 and topics[i-1] == 'B']

        # 去掉前 10 批（冷启动效应）
        def trim(lst, start=10):
            return lst[start//2:] if len(lst) > start else lst

        switch_cost_a = (statistics.mean(trim(a_after_b)) /
                         statistics.mean(trim(a_after_a)) - 1) * 100 \
            if a_after_a else 0
        switch_cost_b = (statistics.mean(trim(b_after_a)) /
                         statistics.mean(trim(b_after_b)) - 1) * 100 \
            if b_after_b else 0

        gain_all = (1 - sum(traj) / sum(frozen)) * 100
        # 分主题增益
        traj_a = [traj[i] for i in range(len(traj)) if topics[i] == 'A']
        frozen_a = [frozen[i] for i in range(len(frozen)) if topics[i] == 'A']
        traj_b = [traj[i] for i in range(len(traj)) if topics[i] == 'B']
        frozen_b = [frozen[i] for i in range(len(frozen)) if topics[i] == 'B']
        gain_a = (1 - sum(traj_a) / sum(frozen_a)) * 100
        gain_b = (1 - sum(traj_b) / sum(frozen_b)) * 100

        results['subjects'][tag] = {
            'online_gain_all': round(gain_all, 1),
            'gain_topic_a': round(gain_a, 1),
            'gain_topic_b': round(gain_b, 1),
            'switch_cost_a_pct': round(switch_cost_a, 1),
            'switch_cost_b_pct': round(switch_cost_b, 1),
            'final_probe_a_gain': probes[-1]['probe_a_gain'],
            'final_probe_b_gain': probes[-1]['probe_b_gain'],
            'final_src_degr': probes[-1]['src_degr'],
            'traj': traj, 'probes': probes}
        print(f'\n=> {tag}: 全程 {gain_all:+.1f}% | A 域 {gain_a:+.1f}% | '
              f'B 域 {gain_b:+.1f}% | 切换成本 A:{switch_cost_a:+.1f}% '
              f'B:{switch_cost_b:+.1f}%\n', flush=True)

        json.dump(results, open(OUT / 'a4_alt_stream.json', 'w'),
                  indent=2, ensure_ascii=False)
        del m; torch.cuda.empty_cache()

    print(f'输出: {OUT / "a4_alt_stream.json"}')


if __name__ == '__main__':
    main()
