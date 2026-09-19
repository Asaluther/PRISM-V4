#!/usr/bin/env python3
"""abl16_8_3seed — 16+8@头2e-4 终端候选的三 seed 确认

单 seed 发现（PHASE2 3b 节）：16+8@2e-4 预训练 59.86（比 12+12@2e-4 好 16 个
PPL 点）而适应持平（流式 +74.2/+59.1 vs +72.0/+60.1）——若 3-seed 复现，
按帕累托接替终端默认。

判读（预注册）：
  ① 预训练：3-seed 均值显著低于 h2e4 的 75.91（非重叠带）
  ② 单发：≥ h2e4 的 +64.3±2.3 误差带下沿
  ③ 流式：@5e-5 与 @1e-4 均不劣于 h2e4（+72.0±2.0 / +60.1±0.4）
协议与 phase2/abl16_8_eval 完全一致（冻结全部 TF 骨干；单发 300 步 5e-4；
流式 STEPS_PER_BATCH @ {5e-5, 1e-4}）。s0 数字引用 abl16_8_h2e4.json。

用法：.venv/Scripts/python.exe abl16_8_3seed.py   (需 GPU)
输出：results/v7/abl16_8_3seed.json
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

CKPTS = {0: 'results/wt_097m_hyb16_8_h2e-4_s0/best_model.pt',
         1: 'results/wt_097m_hyb16_8_h2e-4_s1/best_model.pt',
         2: 'results/wt_097m_hyb16_8_h2e-4_s2/best_model.pt'}
JSONS = {0: 'results/wt_097m_hyb16_8_h2e-4_s0/results.json',
         1: 'results/wt_097m_hyb16_8_h2e-4_s1/results.json',
         2: 'results/wt_097m_hyb16_8_h2e-4_s2/results.json'}


def load_abl(seed):
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=16, n_pcn_layers=8,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPTS[seed], map_location=DEV, weights_only=True))
    return m.to(DEV)


def mean_std(xs):
    return (round(statistics.mean(xs), 2),
            round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0)


def main():
    print('===== 16+8@头2e-4 三 seed 确认 =====\n')
    results = {'ref_h2e4': {'ppl': 75.91, 'ppl_std': 1.02, 'cold': 64.27,
                            'cold_std': 2.25, 'st5e5': 72.0, 'st5e5_std': 2.01,
                            'st1e4': 60.13, 'st1e4_std': 0.4}}

    # ---- 预训练 ----
    ppls = {}
    for s, p in JSONS.items():
        if Path(p).exists():
            ppls[s] = round(json.load(open(p))['best_ppl'], 2)
        else:
            print(f'  [缺] {p}')
    if ppls:
        m, sd = mean_std(list(ppls.values()))
        results['pretrain_ppl'] = {'per_seed': ppls, 'mean': m, 'std': sd}
        print(f'预训练: {ppls} -> {m} ± {sd}  (h2e4: 75.91 ± 1.02)')

    # ---- 单发 ----
    print('\n--- 单发冷启动（3 用户/seed）---')
    users, tests = make_data()
    cold = {}
    for s in (0, 1, 2):
        if not Path(CKPTS[s]).exists():
            continue
        imps = []
        for u in range(len(users)):
            m_ = load_abl(s)
            pb = eval_ppl(m_, tests[u])
            m_ft = finetune(m_, users[u])
            pa = eval_ppl(m_ft, tests[u])
            imps.append((1 - pa / pb) * 100)
        cold[s] = round(statistics.mean(imps), 1)
        print(f'  s{s}: {cold[s]:+.1f}%')
        torch.cuda.empty_cache()
    if len(cold) == 3:
        m, sd = mean_std(list(cold.values()))
        results['coldstart_pct'] = {'per_seed': cold, 'mean': m, 'std': sd}
        print(f'  -> {m} ± {sd}  (h2e4: +64.3 ± 2.3)')

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
    for s in (0, 1, 2):
        if not Path(CKPTS[s]).exists():
            continue
        if s == 0:  # s0 引用已归档评估，避免重复
            prev = json.load(open(OUT / 'abl16_8_h2e4.json'))
            for lr in ST_LRS:
                r = prev[f'stream_lr{lr:g}']
                streaming[f's0_lr{lr:g}'] = {
                    'online_gain_pct': r['online_gain_pct'],
                    'heldout_gain_pct': r['heldout_gain_pct'],
                    'cum_dw': r['cum_dw']}
                print(f'  s0（引用） lr={lr:g}: 在线 {r["online_gain_pct"]:+.1f}%')
            continue
        m_frozen = load_abl(s)
        frozen = [eval_ppl(m_frozen, b) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held)
        del m_frozen; torch.cuda.empty_cache()
        for lr in ST_LRS:
            m_ = load_abl(s)
            traj, held_a, dw = run_stream(m_, stream, user_held, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_gain = (1 - held_a / held_frozen) * 100
            streaming[f's{s}_lr{lr:g}'] = {'online_gain_pct': round(gain, 1),
                                           'heldout_gain_pct': round(held_gain, 1),
                                           'cum_dw': dw}
            print(f'  s{s} lr={lr:g}: 在线 {gain:+.1f}%  held-out {held_gain:+.1f}%  ΔW {dw}')
            del m_; torch.cuda.empty_cache()
    results['streaming'] = streaming
    for lr in ST_LRS:
        vals = [streaming[f's{s}_lr{lr:g}']['online_gain_pct']
                for s in (0, 1, 2) if f's{s}_lr{lr:g}' in streaming]
        if len(vals) == 3:
            m, sd = mean_std(vals)
            results[f'stream_lr{lr:g}_summary'] = {'mean': m, 'std': sd}
            print(f'  @lr={lr:g}: {m} ± {sd}')

    json.dump(results, open(OUT / 'abl16_8_3seed.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "abl16_8_3seed.json"}')


if __name__ == '__main__':
    main()
