#!/usr/bin/env python3
"""T1a — 门控数学等价性对照测试

原式（chunk 循环）：g_ij = sigmoid(w_g · [e_i⊙e_j, e_i+e_j])
等价展开：          logit_ij = q_i·k_j + row_bias_i + col_bias_j
  其中 q_i = w_a⊙e_i, k_j = e_j, row_bias_i = w_b·e_i, col_bias_j = w_b·e_j
  （w_g 拆为 [w_a; w_b] 两半）

判据：同权重同输入下，新旧 logits 的 max |diff| < 1e-5（fp32）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

torch.manual_seed(42)
B, N, D, DG = 4, 128, 64, 16

# 构造与 PCNLayer 相同的门控参数
w_g = torch.randn(1, 2 * DG) * 0.1   # Linear(dg*2, 1, bias=False).weight 形状 [1, 2dg]
e_comp = torch.randn(B, N, DG)
causal = torch.tril(torch.ones(N, N, dtype=torch.bool))

# ---- 原式（chunk 循环，模拟现有实现）----
gates_old = []
for i0 in range(0, N, 128):
    i1 = min(i0 + 128, N)
    e_i = e_comp[:, i0:i1, :].unsqueeze(2)      # [B,c,1,dg]
    e_j = e_comp.unsqueeze(1)                   # [B,1,N,dg]
    inter = torch.cat([e_i * e_j, e_i + e_j], -1)
    gates_old.append(torch.sigmoid((inter @ w_g.T).squeeze(-1)))
gate_old = torch.cat(gates_old, 1).masked_fill(~causal, 0.0)

# ---- 新式（bmm + 偏置）----
w_a, w_b = w_g[:, :DG].squeeze(0), w_g[:, DG:].squeeze(0)   # [dg]
q = e_comp * w_a                                             # [B,N,dg]
k = e_comp
logits = q @ k.transpose(-1, -2)                             # [B,N,N]
logits = logits + (e_comp @ w_b).unsqueeze(-1)               # row bias [B,N,1]
logits = logits + (e_comp @ w_b).unsqueeze(-2)               # col bias [B,1,N]
gate_new = torch.sigmoid(logits).masked_fill(~causal, 0.0)

diff = (gate_old - gate_new).abs().max().item()
print(f'gate max |diff| = {diff:.2e}')
print(f'等价判定: {"✅ PASS (<1e-5)" if diff < 1e-5 else "❌ FAIL"}')

# logits 层面的对照（sigmoid 前，更严格）
logits_old = torch.cat([torch.cat([(torch.cat([e_comp[:, i0:i1].unsqueeze(2) * e_comp.unsqueeze(1),
                                               e_comp[:, i0:i1].unsqueeze(2) + e_comp.unsqueeze(1)], -1)
                                   @ w_g.T).squeeze(-1) for i0 in range(0, N, 128)], 1)])
logits_new = logits
ldiff = (logits_old - logits_new).abs().max().item()
print(f'logits max |diff| = {ldiff:.2e}  {"✅" if ldiff < 1e-4 else "❌"}')
