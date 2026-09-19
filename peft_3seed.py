#!/usr/bin/env python3
"""peft_3seed — PEFT 基线的 3-seed + LoRA init 方差量化 + bitfit 流式补齐

对论文 §5.2 的三个已知缺口：
  ① 仅 seed 0 检查点 → 扩到 s0/s1/s2（tf: wt_200m_tf_lr5e-4_{ckpt,s1,s2}；
     hyb: wt_094m_hyb_{s0,s1,s2}）
  ② LoRA adapter-init 未播种（-26.5% vs -0.4% 重跑方差）→ 每格 3 个固定
     init 抽样（torch.manual_seed 于 setup_peft 前），报 mean±std
  ③ bitfit 流式未测 → 补 {tf,hyb} × lr{5e-5,1e-4} × 3 seed

协议与 peft_baselines 完全一致（adapt 300 步 wd=0、流式 8 批×60 步、
scope: tf=顶 12 块 / hyb=PCN 头）。机制函数全部 import 复用，不改动冻结脚本。

用法：.venv/Scripts/python.exe peft_3seed.py   (需 GPU)
输出：results/v7/peft_3seed.json（新文件，不覆盖 s0 版；逐格增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from coldstart_090m import VOCAB, DEV, OUT, make_data
from peft_baselines import (setup_peft, eval_ppl, adapt, adapter_dw,
                            forward)
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel

CKPTS = {'tf':  ['results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
                 'results/wt_200m_tf_lr5e-4_s1/best_model.pt',
                 'results/wt_200m_tf_lr5e-4_s2/best_model.pt'],
         'hyb': ['results/wt_094m_hyb_s0/best_model.pt',
                 'results/wt_094m_hyb_s1/best_model.pt',
                 'results/wt_094m_hyb_s2/best_model.pt']}
INIT_DRAWS = [101, 202, 303]   # LoRA A 初始化抽样种子


def load_base(kind, seed):
    if kind == 'hyb':
        m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                        n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                        d_gate=128, dropout=0.0, max_seq_len=256)
    else:
        m = TransformerModel(vocab_size=VOCAB, d_model=512, n_layers=24,
                             n_heads=4, ffn_dim=2048, dropout=0.0,
                             max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load(CKPTS[kind][seed], map_location=DEV,
                                 weights_only=True))
    return m.to(DEV)


def mean_std(xs):
    return (round(statistics.mean(xs), 2),
            round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0)


def coldstart_cell(kind, method, rank, lr, seed, users, tests, draws):
    """一格 (kind, method, lr, seed)；LoRA 跑 draws 个固定 init，bitfit 单次"""
    avgs = []
    per_user_last = None
    for draw in (draws if method == 'lora' else [None]):
        imps = []
        for u in range(len(users)):
            m = load_base(kind, seed)
            if draw is not None:
                torch.manual_seed(draw)   # 控制 LoRALinear A 的 randn
            trainable, _ = setup_peft(m, kind, method, rank or 8)
            pb = eval_ppl(m, tests[u], kind)
            m = adapt(m, kind, users[u], trainable, 300, lr)
            pa = eval_ppl(m, tests[u], kind)
            imps.append((1 - pa / pb) * 100)
            del m; torch.cuda.empty_cache()
        avgs.append(round(statistics.mean(imps), 1))
        per_user_last = [round(x, 1) for x in imps]
    m_, sd_ = mean_std(avgs)
    return {'draws': avgs, 'mean': m_, 'std': sd_,
            'per_user_last': per_user_last}


def stream_cell(kind, method, lr, seed, stream, user_held, draw):
    """流式一格：冻结基线参照 + 8 批×60 步 prequential"""
    m_frozen = load_base(kind, seed)
    frozen = [eval_ppl(m_frozen, b, kind) for b in stream]
    held_frozen = eval_ppl(m_frozen, user_held, kind)
    del m_frozen; torch.cuda.empty_cache()

    m = load_base(kind, seed)
    if draw is not None:
        torch.manual_seed(draw)
    trainable, _ = setup_peft(m, kind, method, 8)
    traj = []
    for i, batch in enumerate(stream):
        traj.append(round(eval_ppl(m, batch, kind), 2))
        if i < len(stream) - 1:
            adapt(m, kind, batch, trainable, 60, lr)
    gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
    held_a = eval_ppl(m, user_held, kind)
    held_gain = (1 - held_a / held_frozen) * 100
    dw = adapter_dw(m)
    del m; torch.cuda.empty_cache()
    return {'online_gain_pct': round(gain, 1),
            'heldout_gain_pct': round(held_gain, 1), 'dw_proxy': dw}


def main():
    print('===== PEFT 3-seed + init 方差 + bitfit 流式 =====\n')
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(results, open(OUT / 'peft_3seed.json', 'w'),
                  indent=2, ensure_ascii=False)

    users, tests = make_data()

    # ---- 单发：kind × method × lr × seed（LoRA ×3 init）----
    print('--- 单发冷启动 ---')
    for kind in ('tf', 'hyb'):
        for method in ('lora', 'bitfit'):
            for lr in (5e-4, 1e-3):
                cells = []
                for seed in (0, 1, 2):
                    c = coldstart_cell(kind, method, 8, lr, seed,
                                       users, tests, INIT_DRAWS)
                    cells.append(c)
                key = f'{kind}/{method}@lr{lr:g}'
                all_draws = [d for c in cells for d in c['draws']]
                m_, sd_ = mean_std(all_draws)
                results[key] = {'per_seed': cells, 'mean': m_, 'std': sd_}
                seed_means = [c['mean'] for c in cells]
                print(f'  {key:24s} {m_:+7.1f} ± {sd_:4.1f}  '
                      f'(逐seed {seed_means}, LoRA init 带宽 '
                      f'{min(all_draws):+.1f}~{max(all_draws):+.1f})')
                save()

    # ---- TF rank 扫描 @5e-4 × 3 seed × 3 init ----
    print('\n--- TF LoRA rank 扫描 @5e-4 ---')
    for rank in (4, 16):
        cells = []
        for seed in (0, 1, 2):
            cells.append(coldstart_cell('tf', 'lora', rank, 5e-4, seed,
                                        users, tests, INIT_DRAWS))
        all_draws = [d for c in cells for d in c['draws']]
        m_, sd_ = mean_std(all_draws)
        results[f'tf/lora/r{rank}@lr5e-4'] = {'per_seed': cells,
                                              'mean': m_, 'std': sd_}
        print(f'  r{rank:<2d}: {m_:+.1f} ± {sd_}  (逐seed '
              f'{[c["mean"] for c in cells]})')
        save()

    # ---- 流式：{lora,bitfit} × {tf,hyb} × lr × 3 seed（补 bitfit 空白）----
    print('\n--- 流式（8 批×60 步 prequential）---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained(
        str(Path(__file__).parent / 'tokenizer'))
    from datasets import load_dataset as ld
    ds = ld('roneneldan/TinyStories', split='train')
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    for kind in ('tf', 'hyb'):
        for method in ('lora', 'bitfit'):
            for lr in (5e-5, 1e-4):
                cells = []
                for seed in (0, 1, 2):
                    c = stream_cell(kind, method, lr, seed, stream,
                                    user_held, 1000 + seed)
                    cells.append(c)
                key = f'stream/{kind}/{method}@lr{lr:g}'
                m_, sd_ = mean_std([c['online_gain_pct'] for c in cells])
                results[key] = {'per_seed': cells,
                                'online_mean': m_, 'online_std': sd_}
                print(f'  {key:30s} 在线 {m_:+6.1f} ± {sd_:4.1f}  '
                      f'(held-out {[c["heldout_gain_pct"] for c in cells]})')
                save()

    save()
    print(f'\n输出: {OUT / "peft_3seed.json"}')


if __name__ == '__main__':
    main()
