#!/usr/bin/env python3
"""adapt330_multidraw — 330M 单发冷启动多抽签协议（B1）

方法学缺口：330M 检查点的冷启动用户抽签方差比 96M 大一个量级（同检查点
三次测量 +47.9 / -0.0 / +33.9）——3 用户抽样的均值不稳定。本脚本对 6 个
主线主体（hyb_ng ×3 + tf ×3）各测 **10 个用户抽签**（取 loader 前 20 个
batch 构造），协议不变（300 步 @5e-4，冻结 TF 骨干），产出逐主体
mean ± std over draws，加固论文 §5.1 的 330M 单发数字。

用法：.venv/Scripts/python.exe adapt330_multidraw.py   (需 GPU)
输出：results/v7/adapt330_multidraw.json（逐主体增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import VOCAB, DEV, OUT
from src.data.tinystories import get_dataloaders
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel

KW_NG = dict(topk=128, no_feedback=False, no_gating=True,
             feedback_mode='prev_layer', n_pass=1)
SUBJECTS = [
    ('hyb_ng_s0', 'results/wt_330m_hyb_ng_lr5e-4_s0/best_model.pt', KW_NG),
    ('hyb_ng_s1', 'results/wt_330m_hyb_ng_lr5e-4_s1/best_model.pt', KW_NG),
    ('hyb_ng_s2', 'results/wt_330m_hyb_ng_lr5e-4_s2/best_model.pt', KW_NG),
    ('tf_s0', 'results/wt_353m_tf_lr5e-4_s0/best_model.pt', None),
    ('tf_s1', 'results/wt_353m_tf_lr5e-4_s1/best_model.pt', None),
    ('tf_s2', 'results/wt_353m_tf_lr5e-4_s2/best_model.pt', None),
]
N_USERS = 10
FREEZE_LAYERS = 12


def load_model(tag, ckpt):
    if tag.startswith('hyb'):
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


def coldstart(model, train_data, kw, is_hyb):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    n_freeze = model.layers if is_hyb else model.layers[:FREEZE_LAYERS]
    for blk in n_freeze:
        for p in blk.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
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


def main():
    print('===== 330M 单发多抽签（10 用户 × 6 主体）=====')
    print(f'torch threads ok; N_USERS={N_USERS}\n')

    # 构造 10 用户：取 loader 前 20 batch（偶数 batch 为训练、奇数为测试）
    torch.manual_seed(123)   # 固定抽签种子，保证本协议自身可复现
    train_loader, _, _ = get_dataloaders(seq_len=256, batch_size=32,
                                         num_workers=0, max_train=50000,
                                         max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 2 * N_USERS:
            break
        batches.append((x, y))
    users, tests = [], []
    for u in range(N_USERS):
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
        tests.append({'x': batches[u*2+1][0][:2], 'y': batches[u*2+1][1][:2]})

    results = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, ckpt, kw in SUBJECTS:
        if not Path(ckpt).exists():
            print(f'  [缺] {ckpt} — 跳过')
            continue
        is_hyb = tag.startswith('hyb')
        imps = []
        for u in range(N_USERS):
            m = load_model(tag, ckpt)
            pb = eval_ppl(m, tests[u], kw)
            m = coldstart(m, users[u], kw, is_hyb)
            pa = eval_ppl(m, tests[u], kw)
            imps.append(round((1 - pa / pb) * 100, 1))
            del m; torch.cuda.empty_cache()
        mean = statistics.mean(imps)
        std = statistics.stdev(imps)
        n_pos = sum(1 for x in imps if x > 0)
        results[tag] = {'per_user': imps, 'mean': round(mean, 1),
                        'std': round(std, 1), 'n_positive': n_pos,
                        'n_total': N_USERS}
        print(f'  {tag:10s} {mean:+6.1f} ± {std:4.1f}%  '
              f'(正用户 {n_pos}/{N_USERS})  {imps}')
        json.dump(results, open(OUT / 'adapt330_multidraw.json', 'w'),
                  indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT / "adapt330_multidraw.json"}')


if __name__ == '__main__':
    main()
