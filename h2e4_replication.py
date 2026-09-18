#!/usr/bin/env python3
"""h2e4_replication — 终端候选配置（12+12 头 2e-4）的 3-seed 确认

阶段②遗留缺口：hyb_h2e4 全部结论（预训练 75.66 / 单发 +62.5% / 流式@5e-5 +71.1% /
@1e-4 +60.2% 阈值消失）均为单 seed。本脚本对 s1/s2 新检查点复测同协议后合并
3-seed 统计，判定：
  ① 预训练劣化稳定 ~20%（75±? vs 基线 63.18±0.22）
  ② 单发与基线误差条关系（62.5 vs 60.3±4.3——是否仍在带内）
  ③ 流式优势 + 阈值鲁棒跨 seed 复现（@1e-4 仍显著为正）

协议与 phase2_protocols 完全一致（冻结=全部 TF 骨干；单发 300 步 5e-4；
流式 STEPS_PER_BATCH @ {5e-5, 1e-4}）。s0 数字直接引用 phase2_protocols.json。

用法：.venv/Scripts/python.exe h2e4_replication.py   (需 GPU)
输出：results/v7/h2e4_replication.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from coldstart_090m import VOCAB, DEV, OUT, make_data
from phase2_protocols import eval_ppl, finetune, run_stream, ST_LRS
from src.models.hybrid import HybridModel
from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B

CKPTS = {0: 'results/wt_096m_hyb_h2e-4_s0/best_model.pt',
         1: 'results/wt_096m_hyb_h2e-4_s1/best_model.pt',
         2: 'results/wt_096m_hyb_h2e-4_s2/best_model.pt'}
JSONS = {0: 'results/wt_096m_hyb_h2e-4_s0/results.json',
         1: 'results/wt_096m_hyb_h2e-4_s1/results.json',
         2: 'results/wt_096m_hyb_h2e-4_s2/results.json'}


def load_h2e4(seed):
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12, n_pcn_layers=12,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPTS[seed], map_location=DEV, weights_only=True))
    return m.to(DEV)


def mean_std(xs):
    return (round(statistics.mean(xs), 2),
            round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0)


def main():
    print('===== h2e4（12+12 头 2e-4）3-seed 确认 =====\n')
    results = {}

    # ---- 预训练 PPL ----
    print('--- 预训练 PPL ---')
    ppls = {}
    for s, p in JSONS.items():
        if Path(p).exists():
            ppls[s] = round(json.load(open(p))['best_ppl'], 2)
        else:
            print(f'  [缺] {p}')
    if ppls:
        m, sd = mean_std(list(ppls.values()))
        results['pretrain_ppl'] = {'per_seed': ppls, 'mean': m, 'std': sd}
        print(f'  {ppls} -> {m} ± {sd}  (基线 63.18 ± 0.22, 劣化 '
              f'{round((m / 63.18 - 1) * 100, 1)}%)')

    # ---- 单发冷启动（s0 引用 phase2, s1/s2 实测）----
    print('\n--- 单发冷启动（3 用户/seed）---')
    users, tests = make_data()
    cold = {}
    for s in (1, 2):
        if not Path(CKPTS[s]).exists():
            print(f'  [缺] {CKPTS[s]}')
            continue
        imps = []
        for u in range(len(users)):
            m_ = load_h2e4(s)
            pb = eval_ppl(m_, tests[u])
            m_ft = finetune(m_, users[u])
            pa = eval_ppl(m_ft, tests[u])
            imps.append((1 - pa / pb) * 100)
        cold[s] = round(statistics.mean(imps), 1)
        print(f'  s{s}: {cold[s]:+.1f}%')
        torch.cuda.empty_cache()
    if 'coldstart_pct' in json.load(open(OUT / 'phase2_protocols.json'))['hyb_h2e4']:
        cold[0] = json.load(open(OUT / 'phase2_protocols.json'))[
            'hyb_h2e4']['coldstart_pct']
        print(f'  s0（引用 phase2）: {cold[0]:+.1f}%')
    if len(cold) == 3:
        m, sd = mean_std(list(cold.values()))
        results['coldstart_pct'] = {'per_seed': cold, 'mean': m, 'std': sd}
        print(f'  -> {m} ± {sd}  (基线 60.3 ± 4.3)')

    # ---- 流式 ----
    print('\n--- 流式（prequential, 8 batch）---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    streaming = {}
    for s in (1, 2):
        if not Path(CKPTS[s]).exists():
            continue
        m_frozen = load_h2e4(s)
        frozen = [eval_ppl(m_frozen, b) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held)
        del m_frozen; torch.cuda.empty_cache()
        for lr in ST_LRS:
            m_ = load_h2e4(s)
            traj, held_a, dw = run_stream(m_, stream, user_held, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_gain = (1 - held_a / held_frozen) * 100
            streaming[f's{s}_lr{lr:g}'] = {'online_gain_pct': round(gain, 1),
                                           'heldout_gain_pct': round(held_gain, 1),
                                           'cum_dw': dw}
            print(f'  s{s} lr={lr:g}: 在线 {gain:+.1f}%  held-out {held_gain:+.1f}%  ΔW {dw}')
            del m_; torch.cuda.empty_cache()

    # s0 引用 phase2
    p2 = json.load(open(OUT / 'phase2_protocols.json'))['hyb_h2e4']
    for lr in ST_LRS:
        r = p2[f'stream_lr{lr:g}']
        streaming[f's0_lr{lr:g}'] = {'online_gain_pct': r['online_gain_pct'],
                                     'heldout_gain_pct': r['heldout_gain_pct'],
                                     'cum_dw': r['cum_dw']}
        print(f'  s0（引用 phase2） lr={lr:g}: 在线 {r["online_gain_pct"]:+.1f}%  '
              f'held-out {r["heldout_gain_pct"]:+.1f}%')

    results['streaming'] = streaming
    for lr in ST_LRS:
        vals = [streaming[f's{s}_lr{lr:g}']['online_gain_pct']
                for s in (0, 1, 2) if f's{s}_lr{lr:g}' in streaming]
        if len(vals) == 3:
            m, sd = mean_std(vals)
            results[f'stream_lr{lr:g}_summary'] = {'mean': m, 'std': sd}
            print(f'  @lr={lr:g}: {m} ± {sd}')

    json.dump(results, open(OUT / 'h2e4_replication.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "h2e4_replication.json"}')


if __name__ == '__main__':
    main()
