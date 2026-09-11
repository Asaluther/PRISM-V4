#!/usr/bin/env python3
"""S5 上下文敏感性复检（因果版）

双向时代：打乱 token 顺序 loss 不变 → 模型不用上下文（纯 unigram）。
因果修复后：若模型真实利用过去上下文，打乱（破坏 n-gram 结构）应显著变差。
作用于真实 TinyStories val 数据（非随机 token）。
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB = 50257


@torch.no_grad()
def check(model, kind, batches, kwargs):
    model.eval()
    lo, ls = 0.0, 0.0
    for x, y in batches:
        logits = model(x, **kwargs)
        l_o = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
        # 打乱输入的 token 顺序（保持同样多 token，破坏上下文结构）
        perm_x = torch.argsort(torch.rand_like(x.float()), dim=1).long()
        xs = torch.gather(x, 1, perm_x)
        logits_s = model(xs, **kwargs)
        l_s = F.cross_entropy(logits_s.reshape(-1, VOCAB), y.reshape(-1)).item()
        lo += l_o
        ls += l_s
    n = len(batches)
    return lo / n, ls / n


def main():
    _, val_loader, _ = get_dataloaders(seq_len=256, batch_size=8, max_train=50000, max_val=5000)
    batches = []
    for i, b in enumerate(val_loader):
        if i >= 4:
            break
        batches.append((b[0].to(DEV), b[1].to(DEV)))

    cks = [
        ('ts_causal_transformer', 'transformer', {}),
        ('ts_causal_no_gating', 'pcn', dict(topk=64, no_gating=True)),
        ('ts_causal_twopass', 'pcn', dict(topk=64, feedback_mode='two_pass')),
    ]
    for name, kind, kwargs in cks:
        ck = Path(f'results/{name}_s0/best_model.pt')
        if not ck.exists():
            print(f'{name}: checkpoint 缺失，跳过')
            continue
        if kind == 'transformer':
            m = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                                 ffn_dim=1024, dropout=0.0, max_seq_len=256)
        else:
            m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                         d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed')
        m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        m = m.to(DEV)
        lo, ls = check(m, kind, batches, kwargs)
        delta = ls - lo
        verdict = '✅ 真实上下文依赖' if delta > 0.3 else ('⚠️ 弱' if delta > 0.05 else '❌ 上下文失明')
        print(f'{name:<28} loss原序={lo:.3f}  loss打乱={ls:.3f}  Δ={delta:+.3f}  {verdict}')


if __name__ == '__main__':
    main()
