#!/usr/bin/env python3
"""sweep_ft_lr — 冷启动微调 lr 敏感性（堵审稿攻击面）

coldstart_090m 主结果用的是协议常数 lr 5e-4（小规模时代设定）。
审稿人可质疑：TF 的负改善只是微调 lr 过热的伪影。
本扫描对称检验双方在 {5e-4, 1e-4, 5e-5} 三档微调 lr 下的表现：
若 TF 在所有 lr 下都无实质正改善、而 PCN 在所有 lr 下都大幅改善，
则「冷启动优势在 90M 保持」对微调超参稳健。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from coldstart_090m import (PCN_KW, VOCAB, DEV, FREEZE_LAYERS, OUT,
                            make_data, build_model, eval_ppl, CKPTS)

FT_LRS = [5e-4, 1e-4, 5e-5]


def finetune_lr(model, train_data, is_pcn, lr):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < FREEZE_LAYERS:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(300):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, **PCN_KW) if is_pcn else model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    users, tests = make_data()
    results = {}
    for tag, ckpt in CKPTS.items():
        is_pcn = tag.startswith('PCN')
        for lr in FT_LRS:
            improvements = []
            for u in range(len(users)):
                m = build_model(is_pcn)
                m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
                m = m.to(DEV)
                pb = eval_ppl(m, tests[u], is_pcn)
                m_ft = finetune_lr(m, users[u], is_pcn, lr)
                pa = eval_ppl(m_ft, tests[u], is_pcn)
                improvements.append((1 - pa / pb) * 100)
            avg = sum(improvements) / len(improvements)
            key = f'{tag} @ft_lr={lr:g}'
            results[key] = {'avg': avg, 'improvements': improvements}
            print(f'  {key:<32} 平均改善 {avg:+7.1f}%  '
                  f'(range {min(improvements):+.1f}% ~ {max(improvements):+.1f}%)')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'coldstart_090m_ftlr.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f'\n输出: {OUT / "coldstart_090m_ftlr.json"}')


if __name__ == '__main__':
    main()
