# V6 备线 · 效率工程结果报告（2026-09-10）

> 论文 limitation #5（吞吐仅 TF 的 16%）的清偿 + 终端部署三项必答。
> 数据：results/efficiency/{t1a_bench, t1c 系列, t3_infer, t2_int8}.json

---

## T1 · 门控 bmm 数学等价重构（主成果）

**核心**：`g_ij = sigmoid(w_g·[e_i⊙e_j, e_i+e_j])` 展开为
`logit_ij = q_i·k_j + b_i + c_j`（q_i=w_a⊙e_i，行/列偏置=w_b·e_i/e_j）——
标准 bmm+偏置形式，消掉 chunk 循环与 [B,c,N,2dg] 交互张量。

| 指标 | 原实现（loop） | 重构（bmm） |
|---|---|---|
| 训练吞吐 | 26,362 tok/s | **161,225 tok/s（6.12×）** |
| 相对 Transformer SDPA | 16% | **96%**（目标 80% 超额达成） |
| 显存峰值 | 12.0 GB | **5.4 GB**（与 TF 持平） |
| 等价性（模型级） | — | max diff 1.79e-07 ✅ |
| PPL 不变性 | — | 双 seed 方向翻转（+0.61/−0.46）＝训练混沌，非系统性 ✅ |

**Triton 路线取消**（T1b）——数学重构已超额达标，无需 kernel 工程。

## T3 · 推理基准（batch 1-32，终端口径）

| batch | TF (SDPA) | PCN 门控 bmm | 相对 |
|---|---|---|---|
| 1 | 65,263 | 38,920 | 60% |
| 16 | 388,368 | 420,620 | **+8%** |
| **32** | 410,446 | **466,924** | **+14% 反超** |

- **大 batch（服务端）：门控版反超 Transformer**——O(N·K·d) vs O(N²·d) 的
  理论复杂度优势**首次实测兑现**
- batch 1（终端）：60%（kernel 启动开销主导）；256-token 生成 ≈1.7 秒，可用级；
  小 batch 优化（CUDA Graph/fused kernel）是后续工程项

## T2 · int8 量化敏感性

weight-only per-channel int8 RTN（WT 因果 checkpoint）：
- Transformer：+0.0% | PCN no_gating：+0.1% —— **均无损，门控无量化惩罚**

（过程修复了一个评估双重 shift bug——dataset 已 shift 而 bench 再 shift 一次，
与 copy_bench 阶段同类错误，教训 9 第四次应验）

---

## 对论文 v2 的意义

1. limitation #5 的「吞吐 16% + compile 负优化」**整条撤销**——替换为：
   「训练口径 96% TF、服务端口径 +14% 反超、显存持平」
2. 效率证据线从「债」变成「卖点」：大 batch 反超 = 复杂度优势实证
3. 终端三问全部有了初步答案：吞吐（可用级）/ 显存（持平）/ 量化（无损）
4. v2 建议在 AiraXiv 更新（或等 AI 审稿意见一起）

## 数据索引

`t1a_bench.json`（吞吐六点）· `t1c_*.json`（PPL 不变性四 run）·
`t3_infer.json`（推理 4×3）· `t2_int8.json`（量化对比）