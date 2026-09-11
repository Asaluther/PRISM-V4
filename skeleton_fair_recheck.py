#!/usr/bin/env python3
"""骨架约束公平基线复核：TF ΔW≈2×PCN 在公平 checkpoint 对上是否成立

背景：skeleton_probe / mechanism_study / S 系列全部用 wt_causal_transformer_s0
（lr 3e-4 次优 TF，WT best_ppl=557）作 TF 侧。公平版 TF（wt_tf_lr1e3_s1，
lr 1e-3，PPL≈215）从未被复测——若权重约束（第二涌现性质）是欠训练 TF 的
伪影，公平对上 ΔW 比应塌缩到 ≈1×；若仍 ≈2×，结论对基线质量稳健。

协议：与 skeleton_probe 完全一致（WT checkpoint → 冻结 emb+前6层 → 300 步
lr 5e-4 微调 TS 数据 → ΔW + held-out PPL 变化）。唯一差异：训练/评估块改用
forgetting_benchmark 的确定性奇幻池（skeleton 用随机 batch，两侧数据相同、
对称公平，文档已注明）。PCN 侧 checkpoint 不变（wt_causal_no_gating_s1，
其 lr 3e-4 本就是最优）。

用法：.venv/Scripts/python.exe skeleton_fair_recheck.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from forgetting_benchmark import (build_topic_pool, chunks_to_xy,
                                  eval_ppl, finetune, DEV, KW_A, KW_B,
                                  N_TRAIN, N_HELD, VOCAB)
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

OUT = Path('results/mechanism')


def main():
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    train, held = chunks_to_xy(pool[:N_TRAIN]), chunks_to_xy(pool[-N_HELD:])

    # (标签, kind, 构建, checkpoint)——PCN 换 s1 与 TF 的 s1 配对
    arms = [
        ('PCN (wt_causal_no_gating_s1)', 'pcn',
         lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed'),
         'results/wt_causal_no_gating_s1/best_model.pt'),
        ('TF-次优 (wt_causal_transformer_s0, PPL 557)', 'tf',
         lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                  n_heads=4, ffn_dim=1024, dropout=0.0,
                                  max_seq_len=256, attn_impl='mha'),
         'results/wt_causal_transformer_s0/best_model.pt'),
        ('TF-公平 (wt_tf_lr1e3_s1, PPL 215)', 'tf',
         lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                  n_heads=4, ffn_dim=1024, dropout=0.0,
                                  max_seq_len=256, attn_impl='sdpa'),
         'results/wt_tf_lr1e3_s1/best_model.pt'),
    ]

    results = []
    for tag, kind, build, ckpt in arms:
        m = build()
        m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
        m = m.to(DEV)
        pb = eval_ppl(m, kind, held)
        m_ft, dw, el = finetune(m, kind, train)
        pa = eval_ppl(m_ft, kind, held)
        ch = (1 - pa / pb) * 100
        print(f'  {tag:<42} dW={dw:.4f}  PPL {pb:.0f}>{pa:.0f} ({ch:+.1f}%)')
        results.append({'tag': tag, 'dw': dw, 'ppl_b': round(pb, 1),
                        'ppl_a': round(pa, 1), 'ch': round(ch, 1)})

    dw_pcn = results[0]['dw']
    print('\n===== 判定 =====')
    for r, name in ((results[1], 'TF-次优'), (results[2], 'TF-公平')):
        ratio = r['dw'] / dw_pcn
        keep = '约束稳健' if ratio > 1.5 else ('约束塌缩——疑欠训练伪影' if ratio < 1.2 else '部分保留')
        print(f'  {name}: TF/PCN ΔW = {ratio:.2f}x  -> {keep}')

    json.dump(results, open(OUT / 'skeleton_fair_recheck.json', 'w'), indent=2,
              ensure_ascii=False)
    print(f'\n输出: {OUT}/skeleton_fair_recheck.json')


if __name__ == '__main__':
    main()
