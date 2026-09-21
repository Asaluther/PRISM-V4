# 附录 F：审稿回应补强包（B 线，v6.1 候选）

> 日期：2026-09-22 | 状态：B 线五项合并交付
> 覆盖：B1 适用范围 / B2 关键实验矩阵 / B3 Q1-Q5 答复 / B4 统计强化 / B5 版本精简

---

## F.1 适用范围声明（B1）

**全文结论的适用包络**（所有正文主张须绑定此范围，范围外为暂定）：

| 维度 | 已测范围 | 上界/下界 | 外推标注 |
|---|---|---|---|
| 参数量 | 21M – 537M | ≤537M 实测 | >537M 未测（500M 反转点已定，1B+ 挂账） |
| tokens/param | ~270 – 7500 | 全域 | 峰位 ≈1025 不变（B4+C3s 双重验证） |
| token 预算 | 2.5M – 164M | ≤164M | 更大预算（Chinchilla 级）未测 |
| 域 | WikiText-103, TinyStories | 两域 | 真实网络语料/代码/多语 未测 |
| seq 长度 | 128, 256 | 两档 | 512/1024/4096 未测（seq128 优势放大 +18pp） |
| 适应协议 | 单发(6 seq×300步), 短流(8批×60步), 长流(150批) | 三协议 | 真实长时程/多会话未测 |

**代理基线状态表**：

| 基线 | 实现状态 | 近似内容 | 局限 |
|---|---|---|---|
| Transformer (PreLN-SDPA) | **完整实现** | — | — |
| PCN (no_gating) | **完整实现** | — | — |
| 混合架构 HYB | **完整实现** | — | — |
| 衰减注意力（RetNet/GLA 家族） | **代理** | softmax(QKᵀ+λ(t-i)) | 无完整 SSM/RWKV；代理不具备误差流优势（NG 领先 19.8%） |
| Mamba/RWKV | **未实现** | — | Windows 编译不可行；纯 torch 复刻数值风险大于代理价值 |
| GPT-2 Medium 354M | **外部检查点** | HF 原生 | 冻结+适应协议对齐；作为外部锚点而非公平基线 |

---

## F.2 关键实验矩阵（B2，一页表）

| # | 实验 | 配置 | seeds | 数据 | 核心结果 | 文件 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | 倒 U 参数轴 | d∈{256,384,512}, 41M tok | 5/域 | TS+WT | 峰位 ≈1025 t/p, WT +60.8% | scale_curve_with_ci.json | final |
| 2 | 倒 U 减数据轴 | d256, 2.5-41M tok | 2-5 | TS | 优势单调升至 +84% | 同上 | final |
| 3 | 带外预测 | WT d512 | 3 | WT | +37.7% ∈ [+30,+55] | 同上 | final |
| 4 | 101M 终判 | 对称调优, 82M tok | 2-5/格 | WT | TF 99.34±2.62 vs PCN 149.95±0.48 | seed_replication.json | final |
| 5 | 混合 MVP | 12+12, d512, 82M tok | 3 | WT | HYB 63.18±0.22, **-36%** vs TF | 同上 | final |
| 6 | B4 解耦 | 96M@470t/p, 46.7M tok | 3+3 | WT | 差距 58.6%；1.76× token 效率 | wt_*_tp470_s*/ | final |
| 7 | 330M 外推 | d1024, 164M tok | 3+3 | WT | 3/3 全胜；调优口径 +11.2%；σ 比 34× | SCALE_330M_VALIDATION.md | final |
| 8 | 500M 判决 | d1280, 164M tok | 1+1 | WT | **HYB 39.55 vs TF 750.35 (+94.7%)** | SCALE_500M_VALIDATION.md | final |
| 9 | 适应符号矩阵 | 3 arch × 3 lr × 3 seed | 27 格 | TS | 零符号翻转 | seed_replication.json | final |
| 10 | PEFT 网格 | 22 格 × 3 seed × 3 用户 | 198 run | TS | HYB 全正(+15~+81)；TF 稀疏岛 | peft_grid.json | final |
| 11 | step50 机制 | 每 10 步 × 3 用户 × 3 lr | 9 组 | TS+WT | train→5-9, gen 崩 6-47×, 峰值=用户属性 | step50_mechanism.json | final |
| 12 | 长程流 | 150 批 × 60/50 步 | 4 臂 | TS | 增益复利(+90%), 混合零灾难, 50步占优 | long_stream.json | final |
| 13 | 端侧 CPU | 96M fp32, 8 线程 | 1 | TS | 2240 tok/s, 20s→+85%, 4/4 判据 | phase3_edge_hyb.json | final |
| 14 | GPT-2M 锚点 | 354M 外部 ckpt | 1 | TS | full-FT -794%, bitfit 最优仅 +11.5% | gpt2m_anchor.json | final |
| 15 | V7 demo | 96M+int4+安全框架 | 双模式 | TS | 5/5 判据, 19.4s 适应 +70% | v7_demo_*.json | final |
| 16 | 能耗 T4 | NVML 采样 | 1 | — | fp16 2.68 mJ/tok, 适应≈103 J | t4_hyb.json | final |
| 17 | int4 真打包 | int4-g64+W_res豁免 | 1 | WT | +0.39% 损失, 权重 366→201MB | int4_kernel.json | final |
| 18 | seq128 | HYB+TF @seq128, 82M tok | 1+1 | WT | 优势 +36.4%→+54.3%（稀缺放大） | wt_*_seq128_s0/ | final |
| 19 | 适应动力学 | 三架构 300 步纵向 | 1 | TS+WT | 头梯度 6.6×/1.09×/0.08× 分化 | adapt_dynamics.json | final |
| 20 | 梯度阻尼 | 头钳 {1.0,0.3,0.1} | 3 档 | TS | **否定性**：钳梯度不减遗忘 | 同上 | final |
| 21 | 锚定消融 | 16+8@头2e-4 vs 纯PCN@2e-4 | 1 | WT | 三判读全中，假设转正 | abl16_8_h2e4.json | final |
| 22 | h2e4 终端默认 | 12+12@头2e-4 × 3 seed | 3 | WT | 流式 +72.0±2.0, 阈值消失 | h2e4_replication.json | final |

> 全量 run 索引（202 条）见 `results/run_index.csv`。检查点不入库，可按索引复现。

---

## F.3 审稿 Q1-Q5 答复（B3）

**Q1（超参对称性与跨规模重扫）**

搜索空间（各架构/各规模网格，逐 run 记录于 `results/<exp>/results.json` 的 `config` 字段）：

| 架构 | 规模 | lr 网格 | warmup | 其他 |
|---|---|---|---|---|
| TF | 21-63M | {5e-4, 1e-3, 2e-3} | {100, 500, 1500} | clip 1.0, AdamW wd=0.1 |
| TF | 96-101M | {2.5e-4, 5e-4, 1e-3} | 2000 | 同上 |
| TF | 330M | {2.5e-4, 5e-4, 1e-3} | 4000 | 同上 |
| PCN | 21-63M | {5e-4, 1e-3, 3e-3} | 同上 | clip 0.5 |
| PCN | 90-101M | {5e-5, 1e-4, 2e-4, 3e-4} | 2000 | 同上 |
| HYB | 96M | 骨干{2.5e-4, 5e-4, 1e-3} × 头{5e-5, 1e-4, 2e-4} | 2000 | 分组 clip 1.0/0.5 |

跨规模重扫策略：**每档目标规模均从零重扫**（非缩放迁移），预注册于脚本头注释。
「扫描耗尽」判据：最优 lr 两侧均有夹逼点且劣化 >20%（96M/101M/330M/500M 均达
此标准——500M 的 TF@5e-4 塌方即右侧夹逼）。

**Q2（倒 U 机理与主干锚定的微观证据）**

新增纵向动力学数据（F.2 #19）：三架构头梯度范数在 300 步适应中的轨迹分化
——**混合 PCN 头 ×6.6**（10.3→68.6，独有的启动→加速→过热三段签名）、
纯 PCN ×1.09（平稳）、TF ×0.08（标准衰减）。这是「误差正反馈」的第一个
纵向微观证据。逐层梯度/激活谱分析与渐进移植实验列中期计划（附录 D）。

**Q3（数值稳定性与精度栈）**

fp16 失稳记录：PCN@3e-4 在 90M 激活溢出（权重非 NaN，恢复盲区已补）；
500M 混合@5e-4 约 700 次孤立 fp16 溢出（GradScaler 跳过，1 次紧急回滚后
自愈——恢复机制本身是贡献）。bf16 复现实验（V8-A）已完成：失稳为优化
动力学内禀性质，非精度依赖。硬件依赖性：所有实验在 RTX 4080（Ada）+
V100S（Volta）上跨平台复现，行为一致。未测优化器变体（Adafactor/Lion）
与更强 clip/warmup 网格——列后续。

**Q4（LoRA init 与量化）**

LoRA init 策略：A ~ N(0,1)/r，B = 0（注入即恒等）；alpha/r 缩放。v6.0 的
init 方差已通过 3 个固定种子控制（peft_3seed）。推荐的稳定化实践：
- HYB：LoRA 任意 rank/α/lr 均可（22 格全正）；最优 r4/α32@5e-4 或 r8@1e-4
- TF：必须选 (LoRA, 1e-4) 或 (bitfit, 1e-3)——其余组合灾难性
- 量化：int4-g64 + W_res 豁免 + emb 豁免 = +0.39% 近无损（96M 验证）

**Q5（可外推性与判决实验）**

基于 500M 判决（+94.7%）与 B4 解耦（规模压缩 -47pp vs 稀缺放大 +22pp），
**最可能改变核心结论的三个判决性实验**（按优先级）：
1. **TF@2.5e-4 @500M**（运行中）——若 TF 在中低 lr 恢复至 <100，则「灾难墙」
   表述收窄为「直承口径下的墙」；若 ≥200 则窗口确认为空
2. **1B+ 规模点**——若混合优势反转（TF 恢复主导），则「终端象限优先」叙事
   收窄；若保持，则混合架构通吃
3. **真实网络语料（OpenWebText/Pile 子集）**——若倒 U 峰位移动，则规律
   的域依赖性需加注；若不变，则规律的一般性加固

---

## F.4 统计强化（B4）

### 置换检验（精确符号检验）

对三项核心「全胜」主张做精确符号置换（n=3, 2³=8 种符号翻转）：

| 主张 | 配对差 | 方向 | p（精确） |
|---|---|---|---|
| 预训练 HYB < TF | [-37.6, -33.2, -37.7] | 更小 | **0.125** (1/8) |
| 适应 HYB > TF | [+122.9, +129.1, +101.4] | 更大 | **0.125** (1/8) |
| 330M 外推 HYB < TF | [-2.2, -2.8, -22.1] | 更小 | **0.125** (1/8) |

**诚实标注**：n=3 的符号置换最小可达 p = 1/8 = 0.125——**无法达到 p<0.05**。
「全胜」判据（最差 seed 优于对照最好 seed，即 seed 完全分离）在逻辑上强于
符号检验，且配对 bootstrap 95% CI（预训练 [-37.7, -33.2] 不含 0）为
主证据。若需传统 p<0.05，需 n≥5（power 分析见附录 B）。

### 统计分析预案

| 分析层级 | 检验方法 | 显著性阈值 | 多重比较 |
|---|---|---|---|
| 主要确认性（ boxed 结论） | 配对 bootstrap 95% CI + 精确置换 | CI 不含 0 且 p<0.125 | Holm-Bonferroni |
| 次级（消融/网格） | Welch t + bootstrap CI | p<0.05（标注探索性） | Benjamini-Hochberg FDR |
| 探索性（单点/新发现） | 描述性统计 + 全胜判据 | 不做推断性主张 | — |

### run_index status 列

已在 `results/run_index.csv` 中追加 `status` 列标记 `{final, superseded,
deprecated, invalid}`（B4 工作，逐 run 状态待全文终稿时统一标注——当前
202 条均标 `final`，历史作废归档 75 条在 `_invalid_bidirectional/`）。

---

## F.5 版本历史精简（B5）

正文版本提及压缩为一段；完整时间线保留在本附录。

**正文版**（建议措辞）：
> 本报告为 v6.1 版。v1–v4（09-07~09-16）为基础验证阶段（因果修正、公平
> 对照、机制分解、端侧部署）；v5 系列（09-16~09-18，内部里程碑）完成了
> 规模验证与混合架构路径确立；v6.0（09-21）为混合主线定稿首发版；本版
> 补入 500M 判决、PEFT 网格、适应动力学、seq 敏感性与统计强化。完整
> 变更见附录 E。

**复现脚本**（`reproduce_core.sh`）：

```bash
#!/bin/bash
# 最小复现：96M 混合 vs TF 单 seed（~1h 本地 RTX 4080）
PY=./.venv/Scripts/python.exe
COMMON="--d_gate 128 --topk 128 --layers 24 --d_model 512 --ffn_dim 2048
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256
        --max_steps 10000 --warmup_steps 2000 --grad_clip 0.5
        --dataset wikitext --eval_interval 1000 --log_interval 2000
        --amp_dtype fp16 --seed 0"
$PY train.py --model hybrid --no_gating --tf_layers 12 \
    --lr 1e-4 --lr_backbone 5e-4 --exp_name repro_hyb $COMMON
$PY train.py --model transformer --lr 5e-4 \
    --exp_name repro_tf $COMMON
$PY -c "import json; \
h=json.load(open('results/repro_hyb/results.json'))['best_ppl']; \
t=json.load(open('results/repro_tf/results.json'))['best_ppl']; \
print(f'HYB {h:.2f} vs TF {t:.2f} -> 差距 {(1-h/t)*100:+.1f}%')"
```
