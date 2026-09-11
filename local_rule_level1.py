#!/usr/bin/env python3
"""PRISM R4b 级别 1 — PCN 结构的 W_td 局部规则（混合学习）

设计（V6_LOCAL_RULE_DESIGN.md 级别 1）：
- 两层简化 PCN，周期预测任务（vocab 32, d 64）
- 对照 A：全部参数 BP
- 实验 B：W_td 用 PC 局部规则 ΔW_td ∝ e0 ⊗ h1（误差×上层激活外积，
  W&B 学习规则形式），其余参数 BP
- 判据：B 的最终 loss ≤ A × 1.2（80% 性能）

夜间自测协议：跑一次，无论成败记录现场，不重试。
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V, D, N = 32, 64, 64
STEPS = 1000
OUT = Path('results/mechanism')


class TwoLayerPCN(nn.Module):
    """简化两层 PCN（无门控——级别 1 隔离门控难点）"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(N, D)
        self.ln0 = nn.LayerNorm(D); self.W_bu0 = nn.Linear(D, D, bias=False)
        self.W_td = nn.Linear(D, D, bias=False)          # 局部规则的目标参数
        self.ln_u0 = nn.LayerNorm(D)
        self.W_up0 = nn.Linear(D, D, bias=False); self.W_res0 = nn.Linear(D, D, bias=False)
        self.ln1 = nn.LayerNorm(D); self.W_bu1 = nn.Linear(D, D, bias=False)
        self.ln_u1 = nn.LayerNorm(D)
        self.W_up1 = nn.Linear(D, D, bias=False); self.W_res1 = nn.Linear(D, D, bias=False)
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.eye_(self.W_res0.weight); nn.init.eye_(self.W_res1.weight)
        nn.init.normal_(self.W_td.weight, std=0.01)
        self.last_e0 = None
        self.last_h1 = None

    def forward(self, ids):
        B, Nn = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(Nn, device=ids.device))
        u0 = self.W_bu0(self.ln0(x))
        h0_mid = F.gelu(self.W_up0(self.ln_u0(u0))) + self.W_res0(x)
        u1 = self.W_bu1(self.ln1(h0_mid))
        h1 = F.gelu(self.W_up1(self.ln_u1(u1))) + self.W_res1(h0_mid)
        # 反馈误差（top-down：h1 预测 u0）——记录供局部规则
        e0 = u0 - self.W_td(h1)
        self.last_e0 = e0.detach()
        self.last_h1 = h1.detach()
        # 状态更新（误差修正进层 0 的输出）
        h0 = h0_mid + self.W_up0(self.ln_u0(e0))
        u1b = self.W_bu1(self.ln1(h0))
        h1b = F.gelu(self.W_up1(self.ln_u1(u1b))) + self.W_res1(h0)
        return self.head(self.ln_out(h1b))


def gen(B):
    pat = torch.randint(0, V, (B, 8))
    return pat.repeat(1, N // 8).to(DEV)


def train(local_td=False, lr=3e-4, lr_td=None):
    lr_td = lr_td if lr_td is not None else lr
    torch.manual_seed(0)
    m = TwoLayerPCN().to(DEV)
    bp_params = [p for n_, p in m.named_parameters() if n_ != 'W_td.weight']
    opt = torch.optim.AdamW(bp_params, lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, STEPS + 1):
        ids = gen(32)
        logits = m(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bp_params, 1.0)
        opt.step()
        if local_td:
            with torch.no_grad():
                e0f = m.last_e0.reshape(-1, D)            # [B*N, D]
                h1f = m.last_h1.reshape(-1, D)
                dW = e0f.transpose(0, 1) @ h1f            # [D, D]
                m.W_td.weight += lr_td * dW / e0f.size(0)
        if step % 200 == 0 or step == STEPS:
            with torch.no_grad():
                ev = gen(64)
                l = F.cross_entropy(m(ev)[:, :-1].reshape(-1, V), ev[:, 1:].reshape(-1))
            curve.append((step, round(l.item(), 4)))
    return curve


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV} | 两层简化 PCN | 周期任务 | {STEPS} 步\n")
    bp = train(local_td=False)
    print(f"  A 全 BP          : {bp}")
    lo = train(local_td=True)
    print(f"  B W_td 局部规则  : {lo}")
    ratio = lo[-1][1] / bp[-1][1]
    verdict = 'PASS' if lo[-1][1] <= bp[-1][1] * 1.2 else 'FAIL'
    print(f"\n  级别 1 判定: B/A = {ratio:.2f} → {verdict}（判据 ≤1.2）")
    with open(OUT / 'local_rule_level1.json', 'w') as f:
        json.dump({'bp': bp, 'local_td': lo, 'ratio': round(ratio, 3), 'verdict': verdict}, f, indent=2)
    print(f"输出: {OUT}/local_rule_level1.json")


if __name__ == '__main__':
    main()
