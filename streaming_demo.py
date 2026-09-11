#!/usr/bin/env python3
"""流式适应 demo（GOAL V6#4 未竟项，V6 关闭后的路线兑现）——「边用边学」prequential 协议

场景：用户数据逐批到达（模拟会话流）。每批 i：
  1) 先评估模型对批 i 的 PPL（在线预测得分——模型还没见过它）
  2) 再对批 i 做一次短适应（60 步 @ lr1e-4，校准自 forgetting_calibration arm2）
  3) 记录：下一批 PPL 轨迹、TS 通用漂移、（跨域流）WT 家域漂移、累积 ΔW
对照：冻结基线（不适应）对同一流的 PPL —— 在线增益 = 1 - mean(适应)/mean(冻结)。

双流：
  cross（主流）：WT 公平 checkpoint 对（wt_no_gating_s1 / wt_tf_lr1e3_s1）→ TS 用户流
  indomain（对照）：TS checkpoint 对（ts_causal_no_gating_s1 / ts_tf_lr1e3_s1）→ TS 用户流
  （域内流按今日校准结论预期无增益——量化「已收敛域内无流式空间」的流式形式）

用户流：奇幻池 chunk 0-7（每人 1×256 token/批，8 批一次性通过，不重复 epoch）；
用户 held-out：池末 4 块（流结束后评「学没学到用户风格」）。

用法：.venv/Scripts/python.exe streaming_demo.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from forgetting_benchmark import (build_topic_pool, chunks_to_xy, eval_ppl,
                                  forward_logits, finetune, DEV, KW_A, KW_B, VOCAB)
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import TinyStoriesDataset

OUT = Path('results/demo')
K_BATCHES, STEPS_PER_BATCH, LR = 8, 60, 1e-4

LOADERS = {
    'pcn': lambda ckpt: _load(PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                       n_heads=4, d_gate=64, dropout=0.0,
                                       max_seq_len=256, init_mode='fixed'), ckpt),
    'tf': lambda ckpt: _load(TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                              n_heads=4, ffn_dim=1024, dropout=0.0,
                                              max_seq_len=256,
                                              attn_impl='sdpa'), ckpt),
}


def _load(m, ckpt):
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


def run_stream(kind, ckpt, stream, gen, user_held, extra_evals, tag):
    """prequential 主循环；stream = [{'x':[1,256],'y':[1,256]}, ...]"""
    model = LOADERS[kind](ckpt)
    traj = []
    pre = {n: p.detach().clone() for n, p in model.named_parameters()
           if p.requires_grad is not None and n.startswith('layers.')}
    for i, batch in enumerate(stream):
        rec = {'batch': i,
               'online_ppl': round(eval_ppl(model, kind, batch), 2),
               'gen_ppl': round(eval_ppl(model, kind, gen), 2)}
        for k, d in extra_evals.items():
            rec[f'{k}_ppl'] = round(eval_ppl(model, kind, d), 2)
        traj.append(rec)
        print(f'    [{tag}] 批{i}: 在线 PPL={rec["online_ppl"]:>8.1f}  '
              f'gen={rec["gen_ppl"]:.1f}'
              + (f'  WT={rec.get("wt_held_ppl", 0):.0f}' if extra_evals else ''))
        if i < len(stream) - 1:  # 最后一批只评不学
            model, dw, _ = finetune(model, kind, batch, steps=STEPS_PER_BATCH, lr=LR)
    # 流结束：累积 ΔW + 用户 held-out（末态 vs 冻结，学没学到用户风格）
    deltas = [(p.detach() - pre[n]).norm().item() for n, p in model.named_parameters()
              if n in pre and int(n.split('.')[1]) >= 6]
    final = {'cum_dw': round(sum(deltas) / len(deltas), 4),
             'user_held_ppl_adapted': round(eval_ppl(model, kind, user_held), 2)}
    return traj, final, model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== 流式适应 demo（prequential 边用边学）=====')
    print(f'协议: {K_BATCHES} 批 × 1×256 token，每批先评后学（{STEPS_PER_BATCH} 步 @ lr{LR}）\n')

    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)

    stream = [chunks_to_xy(pool[i:i + 1]) for i in range(K_BATCHES)]
    user_held = chunks_to_xy(pool[-4:])
    val_ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    import numpy as np
    gen = chunks_to_xy(torch.stack(val_ds.data[10:14]).numpy())
    from src.data.wikitext import WikiTextDataset
    wt = WikiTextDataset(split='train', seq_len=256)
    wt_held = chunks_to_xy(wt.data[6000:6004].astype(np.int64))

    streams = {
        'cross': {'ckpt': {'pcn': 'results/wt_causal_no_gating_s1/best_model.pt',
                           'tf': 'results/wt_tf_lr1e3_s1/best_model.pt'},
                  'extra': {'wt_held': wt_held}},
        'indomain': {'ckpt': {'pcn': 'results/ts_causal_no_gating_s1/best_model.pt',
                              'tf': 'results/ts_tf_lr1e3_s1/best_model.pt'},
                     'extra': {}},
    }

    results = {'protocol': {'k_batches': K_BATCHES, 'steps_per_batch': STEPS_PER_BATCH,
                            'lr': LR, 'note': 'prequential：每批先评（在线分）后学'},
               'streams': {}}
    for sname, cfg in streams.items():
        print(f'--- {sname} 流 ---')
        results['streams'][sname] = {}
        for kind in ('pcn', 'tf'):
            traj, final, m_end = run_stream(kind, cfg['ckpt'][kind], stream, gen,
                                            user_held, cfg['extra'], f'{sname}/{kind}')
            m_frozen = LOADERS[kind](cfg['ckpt'][kind])
            final['user_held_ppl_frozen'] = round(eval_ppl(m_frozen, kind, user_held), 2)
            final['user_held_gain_pct'] = round(
                (1 - final['user_held_ppl_adapted'] / final['user_held_ppl_frozen']) * 100, 1)
            results['streams'][sname][kind] = {'traj': traj, 'final': final}
            print(f'    -> 末态用户 held-out: {final["user_held_ppl_adapted"]:.1f} vs '
                  f'冻结 {final["user_held_ppl_frozen"]:.1f} '
                  f'({final["user_held_gain_pct"]:+.1f}%)  累积ΔW={final["cum_dw"]}')
            del m_end, m_frozen
            torch.cuda.empty_cache()
        # 冻结基线（同流同批，不适应）
        for kind in ('pcn', 'tf'):
            m = LOADERS[kind](cfg['ckpt'][kind])
            frz = [round(eval_ppl(m, kind, b), 2) for b in stream]
            results['streams'][sname][kind]['frozen_ppl'] = frz
            adapted = [t['online_ppl'] for t in results['streams'][sname][kind]['traj']][1:]
            frozen_same = frz[1:]
            gain = (1 - sum(adapted) / sum(frozen_same)) * 100
            results['streams'][sname][kind]['online_gain_pct'] = round(gain, 1)
            print(f'  [{sname}/{kind}] 冻结均值 {sum(frozen_same)/len(frozen_same):.1f} -> '
                  f'在线均值 {sum(adapted)/len(adapted):.1f}  (增益 {gain:+.1f}%)')
        print()

    json.dump(results, open(OUT / 'streaming_demo.json', 'w'), indent=2, ensure_ascii=False)
    print(f'输出: {OUT}/streaming_demo.json')


if __name__ == '__main__':
    main()
