#!/usr/bin/env python3
"""gpt2m_anchor — GPT-2 Medium（354M）外部检查点锚点实验

部署地图的关键问题：「为什么不直接下载一个大模型在端上微调？」

用 HF 原生 GPT2LMHeadModel（openai-community/gpt2-medium，354M，24L/d1024，
与我们的 TF 354M 同形状档；同 GPT-2 tokenizer vocab 50257）在主线同构协议下测：

  - 源域定位：WT/TS 冻结 PPL（背景数字——它在 WebText ~9B tokens 上训练，
    预训练对比非公平也不主张；且 WebText 刻意排除 Wikipedia，WT 对它近乎 held-out）
  - 冷启动（3 用户 × 300 步）：半冻结臂（底 12 层冻结，对齐我们的 TF 协议）
    @5e-4 + bitfit 臂（顶 12 块偏置/LN）@1e-3（我们 TF 系最强 PEFT 配置）
  - 流式（8 批 × 60 步 prequential）@{5e-5, 1e-4}（半冻结臂）

预注册判读：
  A. 外部锚点半冻结冷启动灾难（TF 同型）且 bitfit 可救 → 混合架构的
     「零配置」生态位对外部锚点成立，部署地图加固
  B. bitfit/半冻结都稳健 → 如实收窄零配置主张，部署地图改写（外部锚点
     + 方法选择是终端的强基线）
  两个结果都有价值；对照数字：HYB 96M 冷启动 +60.3±4.3%（full-FT 头）、
  TF 101M -57.5±10.5%、TF+bitfit +74.7±2.4%。

用法：.venv/Scripts/python.exe gpt2m_anchor.py   (需 GPU + 已下载缓存)
输出：results/v7/gpt2m_anchor.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from coldstart_090m import make_data
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('results/v7')
FREEZE_LAYERS = 12   # GPT-2 24L：底 12 冻结，对齐我们的 TF 协议


def load():
    m = GPT2LMHeadModel.from_pretrained('gpt2m')   # 本地文件夹（手动下载）
    return m.to(DEV).eval()


@torch.no_grad()
def eval_ppl(model, data):
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x).logits
        l = F.cross_entropy(logits.reshape(-1, 50257),
                            y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def setup_trainable(model, mode):
    """mode='half': 底12层+emb 冻结，顶12训练 | 'bitfit': 全冻，顶12块 bias/LN 解冻"""
    for p in model.parameters():
        p.requires_grad = False
    h = model.transformer.h
    if mode == 'half':
        for blk in h[FREEZE_LAYERS:]:
            for p in blk.parameters():
                p.requires_grad = True
    elif mode == 'bitfit':
        for blk in h[FREEZE_LAYERS:]:
            for nm, p in blk.named_parameters():
                if nm.endswith('bias') or nm.endswith('.weight') and \
                        'ln' in nm.split('.')[0]:
                    p.requires_grad = True
        for nm, p in model.transformer.ln_f.named_parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def adapt(model, data, mode, steps, lr):
    trainable = setup_trainable(model, mode)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = data['x'].to(DEV), data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1]).logits
        loss = F.cross_entropy(logits.reshape(-1, 50257),
                               ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    print('===== GPT-2 Medium 354M 外部锚点（协议对齐主线）=====')
    results = {}

    m0 = load()
    n_params = sum(p.numel() for p in m0.parameters())
    print(f'参数量: {n_params/1e6:.1f}M | 设备: {DEV}')

    # ---- 源域定位（背景） ----
    users, tests = make_data()
    from src.data.wikitext import WikiTextDataset
    import numpy as np
    wt = WikiTextDataset(split='validation', seq_len=256)
    wt_data = {'x': torch.from_numpy(wt.data[:100, :-1].astype('int64')),
               'y': torch.from_numpy(wt.data[:100, 1:].astype('int64'))}
    ts_bg = {'x': torch.cat([t['x'] for t in tests]),
             'y': torch.cat([t['y'] for t in tests])}
    for tag, d in (('wikitext_val', wt_data), ('tinystories_users', ts_bg)):
        ppl = eval_ppl(m0, d)
        results[f'frozen_ppl_{tag}'] = round(ppl, 2)
        print(f'冻结 PPL @ {tag}: {ppl:.2f}')
    del m0; torch.cuda.empty_cache()

    # ---- 冷启动：半冻结臂 + bitfit 臂 ----
    print('\n--- 冷启动（3 用户 × 300 步）---')
    for mode, lr in (('half', 5e-4), ('bitfit', 1e-3)):
        imps = []
        for u in range(len(users)):
            m = load()
            pb = eval_ppl(m, tests[u])
            m = adapt(m, users[u], mode, 300, lr)
            pa = eval_ppl(m, tests[u])
            imps.append(round((1 - pa / pb) * 100, 1))
            del m; torch.cuda.empty_cache()
        results[f'coldstart_{mode}'] = {'per_user': imps,
                                        'avg': round(statistics.mean(imps), 1)}
        print(f'  {mode:7s} @{lr:g}: {imps} -> 均值 {statistics.mean(imps):+.1f}%')

    # ---- 流式（半冻结臂） ----
    print('\n--- 流式（8 批 × 60 步 prequential）---')
    from transformers import GPT2TokenizerFast as TK
    from datasets import load_dataset
    tok = TK.from_pretrained(str(Path(__file__).parent / 'tokenizer'))
    ds = load_dataset('roneneldan/TinyStories', split='train')
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    for lr in (5e-5, 1e-4):
        m_frozen = load()
        frozen = [eval_ppl(m_frozen, b) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held)
        del m_frozen; torch.cuda.empty_cache()

        m = load()
        traj = []
        for i, batch in enumerate(stream):
            traj.append(round(eval_ppl(m, batch), 2))
            if i < len(stream) - 1:
                m = adapt(m, batch, 'half', 60, lr)
        gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
        held_a = eval_ppl(m, user_held)
        held_gain = (1 - held_a / held_frozen) * 100
        results[f'stream_lr{lr:g}'] = {'online_gain_pct': round(gain, 1),
                                       'heldout_gain_pct': round(held_gain, 1),
                                       'traj': traj}
        print(f'  lr={lr:g}: 在线 {gain:+.1f}%  held-out {held_gain:+.1f}%')
        del m; torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'gpt2m_anchor.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "gpt2m_anchor.json"}')


if __name__ == '__main__':
    main()
