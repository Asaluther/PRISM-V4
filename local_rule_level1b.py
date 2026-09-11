#!/usr/bin/env python3
"""PRISM R4b 级别 1（v2）— 用已验证的 src PCNModel 重跑局部规则对照

v1 教训：手写简化结构基线卡 unigram → 实验无效。
v2 方案：直接用 src.models.pcn.PCNModel(n_layers=2)（修复轮+因果版+玩具全链路
验证过的实现），只把 W_td 换成 W&B 局部规则更新，其余 BP。

- 对照 A：全 BP（有效性自检：最终 loss 必须 < 1.0，否则实验无效）
- 实验 B：所有层的 W_td 用局部规则 ΔW_td ∝ e_l ⊗ h_above_l，其余参数 BP
- 判据：B ≤ A × 1.2 且 A 有效
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V, D, N = 32, 64, 64
STEPS = 1000
OUT = Path('results/mechanism')


def gen(B):
    pat = torch.randint(0, V, (B, 8))
    return pat.repeat(1, N // 8).to(DEV)


def eval_loss(m):
    with torch.no_grad():
        ids = gen(64)
        return F.cross_entropy(m(ids, topk=16)[:, :-1].reshape(-1, V),
                               ids[:, 1:].reshape(-1)).item()


def train(local_td, lr=3e-4, lr_td=None):
    lr_td = lr_td if lr_td is not None else lr
    torch.manual_seed(0)
    m = PCNModel(vocab_size=V, d_model=D, n_layers=2, n_heads=4,
                 d_gate=16, dropout=0.0, max_seq_len=N, init_mode='fixed').to(DEV)
    td_ws = [blk.pcn_layer.W_td.weight for blk in m.layers]
    bp_params = [p for n_, p in m.named_parameters() if not n_.endswith('W_td.weight')]
    opt = torch.optim.AdamW(bp_params, lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        ids = gen(32)
        logits = m(ids, topk=16)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bp_params, 1.0)
        opt.step()
        if local_td:
            with torch.no_grad():
                for blk in m.layers:
                    pc = blk.pcn_layer
                    if pc.last_h_above is not None:
                        e = pc.last_e.reshape(-1, D)
                        ha = pc.last_h_above.reshape(-1, D)
                        dW = e.transpose(0, 1) @ ha / e.size(0)
                        pc.W_td.weight += lr_td * dW
        if step % 200 == 0 or step == STEPS:
            curve.append((step, round(eval_loss(m), 4)))
    return curve


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | src PCNModel 2L-d{D} | 周期任务 | {STEPS} 步\n")
    bp = train(local_td=False)
    print(f"  A 全 BP          : {bp}")
    valid = bp[-1][1] < 1.0
    print(f"  A 有效性自检: {'✅ 学会' if valid else '❌ 未学会 → 实验无效（不判 B）'}")
    result = {'bp': bp, 'valid': valid}
    if valid:
        lo = train(local_td=True)
        print(f"  B W_td 局部规则  : {lo}")
        ratio = lo[-1][1] / bp[-1][1]
        verdict = 'PASS' if lo[-1][1] <= bp[-1][1] * 1.2 else 'FAIL'
        print(f"\n  级别 1 判定: B/A = {ratio:.2f} → {verdict}（判据 ≤1.2）")
        result.update({'local_td': lo, 'ratio': round(ratio, 3), 'verdict': verdict})
    else:
        result['verdict'] = 'INVALID'
    with open(OUT / 'local_rule_level1_v2.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"输出: {OUT}/local_rule_level1_v2.json")


if __name__ == '__main__':
    main()
