#!/usr/bin/env python3
"""gate96_pair — gating@96M 孪生对照的配对适应评估

B2：330M gating 检查点适应全灭（-47.8/-104.5）——是 gating 固有还是规模效应？
训练 wt_096m_hyb_gate_s0（= wt_094m_hyb_s0 配置仅去 --no_gating）后，
在**同一次运行内**评 ng_s0 与 gate_s0（共享同一批用户抽签，消除抽签混淆）。

判读：
  gate 适应也塌（≈ng 的一半以下或为负）→ gating 固有（no_gating 决策强化）
  gate 正常（≈ng）→ 330M 全灭是规模效应（单独记录）

协议与 adapt330/96M 系同构：冷启动 3 用户×300 步@5e-4 冻结 TF 骨干；
流式 8 批×60 步@{5e-5,1e-4}。
用法：.venv/Scripts/python.exe gate96_pair.py   (需 GPU)
输出：results/v7/gate96_pair.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import VOCAB, DEV, OUT, make_data
from src.models.hybrid import HybridModel

KW_NG = dict(topk=128, no_feedback=False, no_gating=True,
             feedback_mode='prev_layer', n_pass=1)
KW_GATE = dict(topk=128, no_feedback=False, no_gating=False,
               feedback_mode='prev_layer', n_pass=1)

SUBJECTS = [
    ('hyb512_ng_s0', 'results/wt_094m_hyb_s0/best_model.pt', KW_NG),
    ('hyb512_gate_s0', 'results/wt_096m_hyb_gate_s0/best_model.pt', KW_GATE),
]
ST_LRS = [5e-5, 1e-4]


def load_model(ckpt):
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def eval_ppl(model, data, kw):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, **kw)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                            reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def freeze(model):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def coldstart(model, train_data, kw):
    trainable = freeze(model)
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(300):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1], **kw)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def adapt_batch(model, batch, kw, steps, lr):
    trainable = freeze(model)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = batch['x'].to(DEV), batch['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1], **kw)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    print('===== gating@96M 孪生配对评估 =====\n')
    users, tests = make_data()
    results = {}

    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    for tag, ckpt, kw in SUBJECTS:
        if not Path(ckpt).exists():
            print(f'  [缺] {ckpt} — 跳过')
            continue
        rec = {}
        imps = []
        for u in range(len(users)):
            m = load_model(ckpt)
            pb = eval_ppl(m, tests[u], kw)
            m = coldstart(m, users[u], kw)
            pa = eval_ppl(m, tests[u], kw)
            imps.append(round((1 - pa / pb) * 100, 1))
            del m; torch.cuda.empty_cache()
        rec['coldstart_pct'] = {'per_user': imps,
                                'avg': round(statistics.mean(imps), 1)}
        print(f'  {tag:14s} 冷启动 {rec["coldstart_pct"]["avg"]:+6.1f}% {imps}',
              end='', flush=True)

        m_frozen = load_model(ckpt)
        frozen = [eval_ppl(m_frozen, b, kw) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held, kw)
        del m_frozen; torch.cuda.empty_cache()
        for lr in ST_LRS:
            m = load_model(ckpt)
            traj = []
            for i, batch in enumerate(stream):
                traj.append(round(eval_ppl(m, batch, kw), 2))
                if i < len(stream) - 1:
                    m = adapt_batch(m, batch, kw, 60, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_a = eval_ppl(m, user_held, kw)
            held_gain = (1 - held_a / held_frozen) * 100
            rec[f'stream_lr{lr:g}'] = {'online_gain_pct': round(gain, 1),
                                       'heldout_gain_pct': round(held_gain, 1)}
            print(f'  |  lr={lr:g} 在线 {gain:+.1f}% held {held_gain:+.1f}%',
                  end='', flush=True)
            del m; torch.cuda.empty_cache()
        print()
        results[tag] = rec
        json.dump(results, open(OUT / 'gate96_pair.json', 'w'),
                  indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT / "gate96_pair.json"}')


if __name__ == '__main__':
    main()
