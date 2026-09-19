#!/usr/bin/env python3
"""adapt330 — 330M 检查点的适应协议复测（冷启动 + 流式）

补上 SCALE_330M_VALIDATION 的缺口：330M 点此前只测了预训练轴。本脚本对
补强后的检查点跑与 96M 系完全同构的协议：

- 冷启动：3 用户 × 300 步 @5e-4，冻结 emb + TF 骨干（hyb: layers 全冻；
  tf: 底 12 冻、顶 12 训练——与 coldstart_090m.finetune 语义一致）
- 流式：8 批 × 60 步 prequential @ {5e-5, 1e-4}，ΔW 计 trainable 部分

主体（no_gating 主线）：hyb_ng s0/s1/s2 + tf s0/s1/s2。
附赠对照：gating 版 hyb s0（no_gating=False 前向分支，量化昨晚配置失误的
适应侧影响）。

d1024 构造 + 每检查点正确的 PCN_KW 是本脚本存在的理由（冻结脚本硬编码
d512/no_gating=True）。独立实现，不改动归档脚本。

用法：.venv/Scripts/python.exe adapt330.py   (需 GPU)
输出：results/v7/adapt330.json（逐检查点增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import VOCAB, DEV, OUT, make_data
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel

KW_NG = dict(topk=128, no_feedback=False, no_gating=True,
             feedback_mode='prev_layer', n_pass=1)
KW_GATE = dict(topk=128, no_feedback=False, no_gating=False,
               feedback_mode='prev_layer', n_pass=1)

SUBJECTS = [
    # (标签, 检查点, 前向kwargs)
    ('hyb_ng_s0', 'results/wt_330m_hyb_ng_lr5e-4_s0/best_model.pt', KW_NG),
    ('hyb_ng_s1', 'results/wt_330m_hyb_ng_lr5e-4_s1/best_model.pt', KW_NG),
    ('hyb_ng_s2', 'results/wt_330m_hyb_ng_lr5e-4_s2/best_model.pt', KW_NG),
    ('tf_s0', 'results/wt_353m_tf_lr5e-4_s0/best_model.pt', None),
    ('tf_s1', 'results/wt_353m_tf_lr5e-4_s1/best_model.pt', None),
    ('tf_s2', 'results/wt_353m_tf_lr5e-4_s2/best_model.pt', None),
    ('hyb_gate_s0', 'results/wt_330m_hyb_lr5e-4_s0/best_model.pt', KW_GATE),
]
ST_LRS = [5e-5, 1e-4]
FREEZE_LAYERS = 12


def load_model(tag, ckpt):
    is_hyb = tag.startswith('hyb')
    if is_hyb:
        m = HybridModel(vocab_size=VOCAB, d_model=1024, n_tf_layers=12,
                        n_pcn_layers=12, n_heads=8, ffn_dim=4096,
                        d_gate=128, dropout=0.0, max_seq_len=256)
    else:
        m = TransformerModel(vocab_size=VOCAB, d_model=1024, n_layers=24,
                             n_heads=8, ffn_dim=4096, dropout=0.0,
                             max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def eval_ppl(model, data, kw):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, **kw) if kw else model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                            reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def freeze(model, is_hyb):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    n_freeze = model.layers if is_hyb else model.layers[:FREEZE_LAYERS]
    for blk in n_freeze:
        for p in blk.parameters():
            p.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def coldstart(model, train_data, test, kw, is_hyb):
    trainable = freeze(model, is_hyb)
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(300):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1], **kw) if kw else model(xs[ci:ci+1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def adapt_batch(model, batch, kw, is_hyb, steps, lr):
    trainable = freeze(model, is_hyb)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = batch['x'].to(DEV), batch['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1], **kw) if kw else model(xs[ci:ci+1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    print('===== 330M 适应协议复测（冷启动 + 流式）=====\n')
    users, tests = make_data()
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(results, open(OUT / 'adapt330.json', 'w'),
                  indent=2, ensure_ascii=False)

    # 流式数据（与 seed_replication 同构）
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
        is_hyb = tag.startswith('hyb')
        rec = {}

        # 冷启动
        imps = []
        for u in range(len(users)):
            m = load_model(tag, ckpt)
            pb = eval_ppl(m, tests[u], kw)
            m = coldstart(m, users[u], tests[u], kw, is_hyb)
            pa = eval_ppl(m, tests[u], kw)
            imps.append(round((1 - pa / pb) * 100, 1))
            del m; torch.cuda.empty_cache()
        rec['coldstart_pct'] = {'per_user': imps,
                                'avg': round(statistics.mean(imps), 1)}
        print(f'  {tag:12s} 冷启动 {rec["coldstart_pct"]["avg"]:+6.1f}% '
              f'{imps}', end='', flush=True)

        # 流式
        m_frozen = load_model(tag, ckpt)
        frozen = [eval_ppl(m_frozen, b, kw) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held, kw)
        del m_frozen; torch.cuda.empty_cache()

        pre = None
        for lr in ST_LRS:
            m = load_model(tag, ckpt)
            if pre is None:
                pre = {n: p.detach().clone() for n, p in m.named_parameters()
                       if n.startswith(('layers.', 'pcn_blocks.', 'ln_out'))}
            trainable_names = set()
            tmodel = freeze(m, is_hyb)
            del tmodel
            for n, p in m.named_parameters():
                if p.requires_grad:
                    trainable_names.add(n)
            traj = []
            for i, batch in enumerate(stream):
                traj.append(round(eval_ppl(m, batch, kw), 2))
                if i < len(stream) - 1:
                    m = adapt_batch(m, batch, kw, is_hyb, 60, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_a = eval_ppl(m, user_held, kw)
            held_gain = (1 - held_a / held_frozen) * 100
            dw = sum((p.detach() - pre[n]).norm().item()
                     for n, p in m.named_parameters() if n in trainable_names)
            rec[f'stream_lr{lr:g}'] = {
                'online_gain_pct': round(gain, 1),
                'heldout_gain_pct': round(held_gain, 1),
                'cum_dw': round(dw, 2)}
            print(f'  |  lr={lr:g} 在线 {gain:+.1f}% held {held_gain:+.1f}%'
                  f' ΔW {dw:.1f}', end='', flush=True)
            del m; torch.cuda.empty_cache()
        print()
        results[tag] = rec
        save()

    print(f'\n输出: {OUT / "adapt330.json"}')


if __name__ == '__main__':
    main()
