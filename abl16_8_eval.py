#!/usr/bin/env python3
"""abl16_8_eval — 主干锚定消融：16TF+8PCN @ 头 2e-4 的适应协议测量

阶段②机制假设：h2e4（12+12@头2e-4）之所以在高 lr 下不崩（对照纯 PCN@2e-4
软发散），是因为 TF 主干「锚定」了误差流。若假设成立，更深主干（16TF）+
更薄头（8PCN）@2e-4 应同样稳定，且复现「头 2e-4 解锁适应」的模式
（对照 hyb16_8@1e-4：单发仅 +18.9%）。

判读：
  ① 预训练稳定（零 NaN 软发散）且 PPL 与 hyb16_8@1e-4（54.42）同带
  ② 单发相对 +18.9% 显著抬升（向 h2e4 的 +62.5% 靠拢）
  ③ 流式 @1e-4 转正（hyb16_8@1e-4 时仅 +3.1%）

用法：.venv/Scripts/python.exe abl16_8_eval.py   (需 GPU)
输出：results/v7/abl16_8_h2e4.json
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

CKPT = 'results/wt_097m_hyb16_8_h2e-4_s0/best_model.pt'
RJSON = 'results/wt_097m_hyb16_8_h2e-4_s0/results.json'
REF = {'hyb16_8@1e-4': {'ppl': 54.42, 'cold': 18.9, 'st5e5': 50.8, 'st1e4': 3.1},
       'h2e4(12+12)@2e-4': {'ppl': 75.66, 'cold': 62.5, 'st5e5': 71.1, 'st1e4': 60.2}}


def load_abl():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=16, n_pcn_layers=8,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True))
    return m.to(DEV)


def main():
    print('===== 消融：16TF+8PCN @ 头 2e-4 =====\n')
    results = {'ref': REF}

    if Path(RJSON).exists():
        r = json.load(open(RJSON))
        results['pretrain_ppl'] = round(r['best_ppl'], 2)
        results['nan_recoveries'] = r['nan_recoveries']
        print(f'预训练: PPL={results["pretrain_ppl"]}  '
              f'NaN恢复={results["nan_recoveries"]}  '
              f'(对照 16_8@1e-4: 54.42 | 12+12@2e-4: 75.66)')

    print('\n--- 单发冷启动（3 用户）---')
    users, tests = make_data()
    imps = []
    for u in range(len(users)):
        m = load_abl()
        pb = eval_ppl(m, tests[u])
        m_ft = finetune(m, users[u])
        pa = eval_ppl(m_ft, tests[u])
        imps.append(round((1 - pa / pb) * 100, 1))
        torch.cuda.empty_cache()
    results['coldstart_pct'] = round(statistics.mean(imps), 1)
    print(f'  {imps} -> {results["coldstart_pct"]:+.1f}%  '
          f'(16_8@1e-4: +18.9% | h2e4: +62.5%)')

    print('\n--- 流式 @ {5e-5, 1e-4} ---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    m_frozen = load_abl()
    frozen = [eval_ppl(m_frozen, b) for b in stream]
    held_frozen = eval_ppl(m_frozen, user_held)
    del m_frozen; torch.cuda.empty_cache()
    for lr in ST_LRS:
        m = load_abl()
        traj, held_a, dw = run_stream(m, stream, user_held, lr)
        gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
        held_gain = (1 - held_a / held_frozen) * 100
        results[f'stream_lr{lr:g}'] = {'online_gain_pct': round(gain, 1),
                                       'heldout_gain_pct': round(held_gain, 1),
                                       'cum_dw': dw}
        print(f'  lr={lr:g}: 在线 {gain:+.1f}%  held-out {held_gain:+.1f}%  ΔW {dw}'
              f'  (16_8@1e-4: {REF["hyb16_8@1e-4"]["st5e5" if lr == 5e-5 else "st1e4"]:+.1f}%)')
        del m; torch.cuda.empty_cache()

    json.dump(results, open(OUT / 'abl16_8_h2e4.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "abl16_8_h2e4.json"}')


if __name__ == '__main__':
    main()
