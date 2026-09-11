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

---

## T4 · 能耗基准（2026-09-11 补，GOAL V5#5 首次执行）

NVML 功率采样（50ms）× 训练/推理两口径，`bench_energy.py` → `t4_energy.json`。
RTX 4080 空闲 35.4W，绝对值不外推手机 SoC，结论看架构间比值。

| 口径 | TF | PCN no_gating | PCN 门控 bmm |
|---|---|---|---|
| 训练 mJ/tok（abs） | 1.253 | 1.386 | **1.148** |
| 推理 b=1 mJ/tok | **1.134** | 2.253 | 2.297 |
| 推理 b=32 mJ/tok | 0.709 | 0.769 | **0.618** |

- **训练/服务端**：门控 bmm 每 token 能耗最低（训练 -8%、b32 推理 -13% vs TF）——
  复杂度优势在能耗口径同样兑现
- **终端 b=1**：PCN 每 token 能耗约为 TF 的 2 倍（吞吐 60% 主导，功率相近）——
  「低功耗终端」论点当前**不成立**，batch1 kernel 优化（CUDA Graph/fused）是前置条件
- **到同 PPL 训练能耗**（组合 fairness 曲线，TS 40.96M tokens）：TF 42kJ @ PPL 15.1 vs
  no_gating 47kJ @ PPL 16.5——充分训练区 TF 占优，与倒 U 右侧一致；低数据区的能耗-性能
  组合（2.5M-10M token 切片）未测，是后续项

---

## T5 · batch1 CUDA Graph 优化（2026-09-11 晚，T4 短板清偿）

**问题**：b1 终端口径 gate_bmm 吞吐仅 TF 的 ~60%、能耗 2×（T4）——12 层 × 每层 15-20 个
微小 kernel 的启动开销主导（d256×N256 单算子过小）。

**手段**（`bench_b1_graph.py` → `t5_b1_graph.json`）：
1. 整前向 CUDA Graph 捕获/重放（静态形状 N=256，side-stream 预热标准流程）
2. `pcn.py` no_gating 路径 MHA `need_weights=False`（`last_attn_w` 全仓零读取方，
   True 强制慢路径）；PPL 回归校验 19.51/14.37 vs 基准 19.5/14.4 ✅

| b1 | eager | graph | 提速 | graph 能耗 | vs TF-graph |
|---|---|---|---|---|---|
| TF(SDPA) | 69,479 | 190,268 | ×2.74 | 0.71 mJ/tok | 100% |
| PCN no_gating | 44,591 | **174,107** | ×3.90 | 0.69 mJ/tok | **91.5%** ✅ |
| PCN gate_bmm | 35,822 | **194,397** | **×5.43** | **0.51 mJ/tok** | **102.2%** ✅ |

b4：no_gating 96.2% / gate_bmm 99.8%，同样达标。数值等价：全部配置
graph vs eager **位级一致**（|Δlogit| = 0）。

**判定**：判据「≥90% TF（优化对优化）」达成——
- 吞吐：b1 60% → 102%（gate_bmm）；256-token 前向 ≈1.3ms
- **能耗：b1 从 2× 劣势反转为 0.51 vs 0.71 mJ/tok（-28%）**——T4 的「低功耗终端
  不成立」结论撤销，改为「graph 化部署下成立」

**边界与后续**：
- graph 需静态形状：自回归逐 token 生成（N 递增）需按长度预捕图或 padding 策略；
  当前 demo 均为全上下文前向，不受影响。部署集成（输入拷入静态缓冲/输出拷出）是工程项
- graph 化同时适用于 TF（×2.74），故比较口径必须是优化对优化——本文全部如此
- eager 口径（无 graph）下 PCN 仍慢于 TF；终端部署应以 graph 模式为准

---

## T6 · int4 权重量化敏感性 + W_res 主干发现（2026-09-11 晚，GOAL V7#2 前置数据）

**协议**（`bench_int4.py` → `t6_int4.json`）：RTN 假量化（精度口径），2D 权重除 embedding，
公平 checkpoint 对（wt_tf_lr1e3_s1 / wt_causal_no_gating_s1 + TS 域内对），双域 50 chunk 评估。

| 配置 | TF（WT/TS） | PCN no_gating（WT/TS） |
|---|---|---|
| int8 对称 per-channel | -0.0% / -0.0% | +0.0% / +0.0%（T2 结论复现） |
| int4 对称 per-channel | +1.8% / +2.5% | **+213% / +167%** |
| int4 非对称 group128 | +0.6% / +0.8% | +19.5% / +15.7% |
| int4 非对称 group64 | +0.1% / +0.6% | +5.0% / +4.3% |
| **int4-g64 + W_res 保 fp16** | — | **+0.2% / +0.4%（近无损）** ~32.0 MB |

**核心发现：PCN 对 int4 的敏感性 ≈ 全部来自 W_res**（探针：仅量化 W_res 一个矩阵
即 +4.2~4.5%，≈ 全模型 int4-g64 的全部损失；W_res 除外后近无损）。

**机制**：PCN 的残差主干 `h = GELU(W_up(·)) + W_res(x)` 中**信号主干本身穿过 W_res
matmul（非加性恒等）**，量化误差直接进入主干并跨 12 层复合；TF 的 `x + branch(x)`
加性残差天然把量化误差隔离在增量支路。这与 V4 诊断「W_res 恒等初始化修复梯度直通」
是**同一设计选择的两面**：训练期信号好（恒等通路），量化期脆弱（主干可扰）。

**结论**：
1. 端侧部署配方：**PCN int4-g64 非对称 + W_res（及 embedding）保 fp16** → 近无损
   （+0.2~0.4%），32.0 MB；TF 可直接 int4-g64（+0.1~0.6%）
2. int8 时代「两者皆无损」在 4bit 断裂：PCN 需混合精度，TF 不需要——
   「PCN 更耐量化」不成立，改为「PCN 的量化边界在残差主干」
3. 体积注意：22M 规模下 embedding 占 61% 参数（fp16 保留），int4 只省线性层；
   V7 的 0.5-2B 目标规模下 embedding 占比下降，int4 收益才充分兑现
4. GPTQ 误差补偿无需立项——分组 + W_res 例外已近无损