#!/usr/bin/env python3
"""streaming_090m — 流式适应协议 90M 版（prequential 边用边学，cross 流）

小规模（12L, streaming_demo.py）：在线增益 PCN +24.0% vs TF -80.9%。
本脚本在 90-101M 对称调优检查点上重跑同一协议。与原版的差异仅规模适配项：
  模型构造 24L/d512/d_gate128(PCN)/ffn2048(TF)；PCN 前向 topk=128 + 完整
  model_kwargs（与训练时一致，见 coldstart_090m.PCN_KW）；冻结底部 12/24 层
  （同原版 6/12 比例）。数据流/评估/ΔW 口径逐项不变。

主协议 lr=1e-4（原版常数）；另跑 5e-5 / 2.5e-4 两档对称迷你扫描，
堵「流式 lr 未在本规模校准」的攻击面。

用法：.venv/Scripts/python.exe streaming_090m.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B, DEV, VOCAB
from coldstart_090m import PCN_KW, build_model   # 90M 构造 + 训练一致的前向 kwargs
from src.data.tinystories import TinyStoriesDataset
from src.data.wikitext import WikiTextDataset

OUT = Path('results/v7')
K_BATCHES, STEPS_PER_BATCH, LR = 8, 60, 1e-4
FREEZE_LAYERS = 12   # 底部一半（同小规模 6/12 的比例）
FT_LRS = [1e-4, 5e-5, 2.5e-4]   # 首项为主协议常数

CKPTS = {
    'pcn': 'results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt',
    'tf': 'results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
    'hyb': 'results/wt_094m_hyb_s0/best_model.pt',
}


def load_ckpt(kind):
    m = build_model(kind)
    m.load_state_dict(torch.load(CKPTS[kind], map_location=DEV, weights_only=True))
    return m.to(DEV)


def forward_logits(model, kind, x):
    return model(x, **PCN_KW) if kind in ('pcn', 'hyb') else model(x)


@torch.no_grad()
def eval_ppl(model, kind, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = forward_logits(model, kind, x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def adapt_batch(model, kind, batch, steps, lr):
    """与 forgetting_benchmark.finetune 同构；每批新建优化器（原版行为）"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < FREEZE_LAYERS:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = batch['x'].to(DEV), batch['y'].to(DEV)
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


def run_stream(kind, stream, gen, user_held, wt_held, lr, tag):
    model = load_ckpt(kind)
    traj = []
    # ΔW 快照覆盖两种命名：纯架构 layers.* / hybrid 的 pcn_blocks.*（可训练部分）
    pre = {n: p.detach().clone() for n, p in model.named_parameters()
           if n.startswith(('layers.', 'pcn_blocks.'))}
    for i, batch in enumerate(stream):
        rec = {'batch': i,
               'online_ppl': round(eval_ppl(model, kind, batch), 2),
               'gen_ppl': round(eval_ppl(model, kind, gen), 2),
               'wt_held_ppl': round(eval_ppl(model, kind, wt_held), 2)}
        traj.append(rec)
        print(f'    [{tag}] 批{i}: 在线 PPL={rec["online_ppl"]:>8.1f}  '
              f'gen={rec["gen_ppl"]:.1f}  WT={rec["wt_held_ppl"]:.0f}')
        if i < len(stream) - 1:   # 最后一批只评不学
            model = adapt_batch(model, kind, batch, STEPS_PER_BATCH, lr)
    # ΔW 只算可训练部分：layers ≥FREEZE_LAYERS（纯架构）/ pcn_blocks.*（hybrid 适应头）
    deltas = [(p.detach() - pre[n]).norm().item() for n, p in model.named_parameters()
              if n in pre and (n.startswith('pcn_blocks.')
                               or int(n.split('.')[1]) >= FREEZE_LAYERS)]
    final = {'cum_dw': round(sum(deltas) / len(deltas), 4),
             'user_held_ppl_adapted': round(eval_ppl(model, kind, user_held), 2)}
    return traj, final


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== 流式适应 90M（prequential 边用边学，cross 流）=====')
    print(f'协议: {K_BATCHES} 批 × 1×256 token，每批先评后学（{STEPS_PER_BATCH} 步），'
          f'冻结 emb+底部{FREEZE_LAYERS}/24 层，fp32，流式 lr 扫描 {FT_LRS}\n')

    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(K_BATCHES)]
    user_held = chunks_to_xy(pool[-4:])
    val_ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    gen = chunks_to_xy(torch.stack(val_ds.data[10:14]).numpy())
    wt = WikiTextDataset(split='train', seq_len=256)
    wt_held = chunks_to_xy(wt.data[6000:6004].astype(np.int64))

    results = {'protocol': {'k_batches': K_BATCHES,
                            'steps_per_batch': STEPS_PER_BATCH,
                            'main_lr': LR, 'ft_lrs': FT_LRS,
                            'freeze_layers': FREEZE_LAYERS,
                            'note': '90M 版，协议同 streaming_demo（cross 流），'
                                    '仅模型构造/冻结层数规模适配'},
               'ckpt': CKPTS, 'streams': {}}

    # 冻结基线与流末 held-out 冻结对照（与 lr 无关，每臂算一次）
    frozen_all, user_held_frozen = {}, {}
    for kind in ('pcn', 'tf', 'hyb'):
        m = load_ckpt(kind)
        frozen_all[kind] = [round(eval_ppl(m, kind, b), 2) for b in stream]
        user_held_frozen[kind] = round(eval_ppl(m, kind, user_held), 2)
        del m; torch.cuda.empty_cache()
        print(f'  [{kind}] 冻结基线 在线均值 '
              f'{sum(frozen_all[kind])/len(frozen_all[kind]):.1f} | '
              f'用户 held-out {user_held_frozen[kind]:.1f}')

    for lr in FT_LRS:
        print(f'\n--- 流式适应 lr={lr:g} ---')
        results['streams'][f'lr={lr:g}'] = {}
        for kind in ('pcn', 'tf', 'hyb'):
            tag = f'{lr:g}/{kind}'
            traj, final = run_stream(kind, stream, gen, user_held, wt_held, lr, tag)
            final['user_held_ppl_frozen'] = user_held_frozen[kind]
            final['user_held_gain_pct'] = round(
                (1 - final['user_held_ppl_adapted'] / user_held_frozen[kind]) * 100, 1)
            adapted = [t['online_ppl'] for t in traj][1:]
            frozen_same = frozen_all[kind][1:]
            gain = (1 - sum(adapted) / sum(frozen_same)) * 100
            final['online_gain_pct'] = round(gain, 1)
            results['streams'][f'lr={lr:g}'][kind] = {'traj': traj, 'final': final,
                                                      'frozen_ppl': frozen_all[kind]}
            print(f'  -> [{kind}] 在线增益 {gain:+.1f}% | 用户 held-out '
                  f'{final["user_held_gain_pct"]:+.1f}% | 累积ΔW={final["cum_dw"]}')
            torch.cuda.empty_cache()

    json.dump(results, open(OUT / 'streaming_090m.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT}/streaming_090m.json')

    print('\n===== 判读（主协议 lr=1e-4）=====')
    for kind in ('pcn', 'tf', 'hyb'):
        f = results['streams']['lr=0.0001'][kind]['final']
        print(f'  {kind}: 在线增益 {f["online_gain_pct"]:+.1f}% | '
              f'held-out {f["user_held_gain_pct"]:+.1f}% | ΔW {f["cum_dw"]}')


if __name__ == '__main__':
    main()
