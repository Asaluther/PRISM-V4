# PRISM V4 — 全新架构方案设计

> 2026-06-24 | 基于 V1-V3 教训（LESSONS.md）从零设计

---

## 目录

1. [文献调研：Transformer 之外的计算范式](#1-文献调研)
2. [成功标准定义](#2-成功标准定义)
3. [三个架构候选](#3-三个架构候选)
4. [推荐方向：预测编码网络（PCN）](#4-推荐方向)
5. [实验设计](#5-实验设计)
6. [风险评估](#6-风险评估)

---

## 1. 文献调研

### 1.1 当前非 Transformer 架构全景

通过调研 2023-2026 年的主要工作，现有 Transformer 替代方案可分为以下几类：

#### A. 状态空间模型（SSM）族

| 架构 | 核心机制 | 优势 | 局限 |
|------|---------|------|------|
| **Mamba** [Gu & Dao, 2023] | 选择性状态空间模型，输入依赖的参数化 | 线性复杂度，内容选择能力强 | 固定大小状态限制精确回忆 |
| **Mamba-2** [Gu et al., 2024] | 结构化状态空间对偶（SSD），建立 SSM 与 attention 的数学联系 | 比 Mamba 快 2-8×，与 attention 有理论统一 | 本质仍是 SSM，同样的状态瓶颈 |
| **Hyena** [Poli et al., 2023] | 隐式参数化长卷积 + 数据控制门控 | 亚二次复杂度，在长序列上优于其他无 attention 模型 | 卷积结构对非平稳语言模式适应性有限 |

**关键发现**：Mamba-2 论文（2405.21060）证明了 SSM 和 attention 通过"半可分矩阵"分解是**数学等价**的。这意味着 SSM 并非真正"不同"的范式——它只是 attention 的一个结构化特例。

#### B. 线性注意力 / 循环网络族

| 架构 | 核心机制 | 优势 | 局限 |
|------|---------|------|------|
| **RWKV** [Peng et al., 2023] | 线性注意力递归，token-shift + 通道混合 | O(1) 推理，Transformer 级质量 | 固定状态，精确回忆能力弱 |
| **RetNet** [Sun et al., 2023] | 保持机制（retention），支持并行/循环/分块三种计算模式 | 理论上统一了 attention 和循环 | 保持机制本质是衰减线性注意力 |
| **Griffin/Hawk** [De et al., 2024] | 门控线性递归 + 局部 attention 混合 | 匹配 Llama-2 性能，外推能力强 | 混合架构仍依赖 attention 做回忆 |

**关键发现**："Repeat After Me" [Jelassi et al., 2024]（2402.01032）从理论上证明：**两层的 Transformer 可以复制指数长度的字符串，而所有固定大小状态的 GSSM（广义状态空间模型）在复制任务上存在根本性限制**。这不是工程问题，是信息论限制。

#### C. 混合架构

| 架构 | 核心机制 | 意义 |
|------|---------|------|
| **BASED** [Arora et al., 2024] | 线性 attention + 滑动窗口 attention 混合 | 识别了"回忆-效率"权衡前沿（recall-throughput tradeoff） |
| **Titans** [Behrouz et al., 2025] | 神经长期记忆模块（test-time learning）+ attention | 引入了"学习记忆"的概念，可扩展到 2M+ 上下文 |
| **Miras** [Behrouz et al., 2025] | 统一框架：联想记忆 + 注意力偏置 + 保持正则化 | 证明大多数模型都可归约为不同的"注意力偏置目标" |

**关键发现**：Miras 论文（2504.13173）揭示了一个深刻的事实——**几乎所有现有序列模型都可以表述为：用某种目标函数学习 key-value 关联的联想记忆模块**。它们的差异仅在于：(1) 联想记忆的结构，(2) "注意力偏置"目标（点积？L2？其他？），(3) 遗忘机制（保持正则化）。

#### D. 其他创新

- **DyT** [Zhu et al., 2025]（2503.10622）：用 `tanh(αx)` 替代 LayerNorm，不是架构创新而是训练技巧
- **GoldFinch** [Goldstein et al., 2024]：RWKV 生成压缩 KV-cache + Transformer，本质是混合
- **Titans 的神经记忆**：在 test-time 执行梯度下降来记忆历史——这是一个新的计算范式

### 1.2 核心结论

> **Transformer 之外已经被验证可行的计算范式只有一种：压缩到固定大小状态的循环模型（SSM/RNN/线性注意力）。但它们在精确回忆（recall）任务上存在根本性信息论限制。**

所有声称"超越 Transformer"的架构要么：
1. **是 Transformer 的数学等价物**（如 Mamba-2 ≈ structured attention）
2. **在 recall 任务上明显弱于 Transformer**（如 Mamba, RWKV）
3. **是混合架构，仍依赖 attention**（如 Griffin, BASED）
4. **引入了新的计算概念但尚未在语言建模上证明**（如 Titans 的神经记忆）

### 1.3 从 V1-V3 教训看 V4 的起点

V1-V3 的核心教训是"从 Transformer 出发做减法 → 回到 Transformer"。V4 需要从一个**不属于 attention-SSM-RNN 谱系**的起点出发。

但文献调研揭示了一个残酷的事实：**所有已知的有效序列建模方法，要么是 attention 的变体，要么在 recall 上有根本限制。**

这意味着 V4 的策略不能是"找一个已知的替代范式然后优化它"，而必须是：

> **从完全不同的计算原理出发，接受可能不如 Transformer 的事实，但证明它拥有 Transformer 不具备的能力。**

---

## 2. 成功标准定义

### 2.1 什么样的结果能证明"这不是 Transformer 的变体"？

**必要条件（全部满足才算成功）：**

1. **计算基元不同**：核心运算不是 QK^T softmax(·) V（attention），不是状态空间递归（SSM），不是线性注意力核分解。这些都有数学等价关系。

2. **可区分的行为签名**：在小规模（10-50M 参数）上，存在至少一个任务/指标，新架构的表现**显著**不同于任何规模的 Transformer——不是"更好"或"更差"，而是表现出**定性的行为差异**。

3. **不可归约为 attention**：在 Miras 框架（2504.13173）下，新架构不能被描述为"某个注意力偏置目标 + 保持门控"的组合。

4. **结构探针差异**：通过 probing 实验证明，新架构学到的内部表示在结构上不同于 Transformer（不仅仅是不同的特征，而是不同类型的表示）。

### 2.2 具体验收指标

| 指标 | Transformer 基线行为 | "不同"的判据 |
|------|---------------------|-------------|
| Perplexity vs 模型大小 | power law scaling | 明显偏离 power law（无论方向） |
| Recall 任务（MQAR） | 接近 100%（n ≤ 64） | 表现出不同的 scaling 曲线 |
| 复制任务 | 完美复制 | 不同的错误模式（非随机） |
| 表示相似性（CKA） | 层间 CKA 呈特定模式 | CKA 模式显著不同 |
| 训练动态 | loss 曲线呈特定形状 | 不同的学习阶段或平台期 |
| 层数 vs 宽度 | compute-optimal 有特定比例 | 最优配置落在不同区域 |

**核心判据：** 不需要在 PPL 上打败 Transformer。需要在**至少两个维度**上表现出定性不同的行为，且这些差异不能用"训练不充分"或"正则化效应"解释。

### 2.3 失败标准（明确什么不算成功）

- ❌ PPL 略好/略差于 Transformer → 不够，这只是调参差异
- ❌ 更快/更省内存 → 不够，这是工程不是架构差异
- ❌ 在特定任务上好 → 不够，可能是归纳偏置的偶然匹配
- ❌ "加了新模块后性能提升" → 不够，这是给 Transformer 加组件

---

## 3. 三个架构候选

### 候选 A：预测编码网络（Predictive Coding Network, PCN）

#### 核心计算基元

**预测误差驱动的层级计算**（Hierarchical Prediction Error Computation）

与传统架构的根本区别在于信息流方向：
- Transformer：**前馈**（bottom-up only），每层对整个表示做变换
- SSM/RNN：**递归**（left-to-right），状态沿时间演化
- **PCN：双向**（bottom-up errors + top-down predictions），每层同时接收来自下层的信号和来自上层的预测

核心运算不是"计算两个 token 的相似度"（attention），也不是"更新时变状态"（SSM），而是：

$$\text{error}_l = \text{input}_l - \text{prediction}_l(\text{state}_{l+1})$$

表示的"内容"是**预测误差**，而非原始特征。只有"意料之外"的信息才向上传播。

#### 架构详述

```
PCN 层（Layer l）
═══════════════════════════════════════════════

输入: x_l ∈ ℝ^{n×d}（来自下层），h_{l+1} ∈ ℝ^{n×d}（来自上层，首轮为零）

步骤 1 — 自底向上特征提取:
  u_l = LayerNorm(W_bu^l · x_l)        ∈ ℝ^{n×d}

步骤 2 — 自顶向下预测:
  p_l = W_td^l · h_{l+1}                ∈ ℝ^{n×d}
  （首层无上层信号时 p_l = 0）

步骤 3 — 预测误差计算:
  e_l = u_l - p_l                        ∈ ℝ^{n×d}

步骤 4 — 误差驱动的跨位置交互:
  门控值: g_{ij} = σ(w_g^T · [e_i ⊙ e_j + e_i + e_j])
  加权误差聚合: ē_i = Σ_j g_{ij} · (W_lat^l · e_j)
  
  注意：这里 g_{ij} 不是基于内容相似度（不同于 attention 的 QK^T），
  而是基于"两个位置的预测误差有多大关联"。
  高误差位置之间的交互被增强，可预测位置之间的交互被抑制。

步骤 5 — 状态更新（残差 + 误差修正）:
  h_l = σ(W_up^l · (e_l + ē_l) + W_res^l · x_l)  ∈ ℝ^{n×d}

输出: h_l → 传递给下层作为 h_{l+1}
      h_l → 传递给上层作为 x_{l+1}
```

#### 为什么它在理论上可能学到不同的东西

1. **稀疏信息传播**：只有预测误差传播，可预测部分被"解释掉"。Transformer 传播所有信息。
2. **内建层级结构**：高层修正低层的预测，自然形成抽象层次。Transformer 的层级是均匀的。
3. **上下文驱动的门控**：位置间交互由预测误差驱动（"惊讶"），而非内容相似度。这导致完全不同的聚合模式。
4. **反馈连接**：高层信息可以影响低层处理。Transformer 是纯前馈的。

#### 参数估算

| 组件 | 参数量 | d=256 | d=384 |
|------|--------|-------|-------|
| W_bu | d² | 65K | 147K |
| W_td | d² | 65K | 147K |
| W_lat | d² | 65K | 147K |
| W_up | d² | 65K | 147K |
| W_res | d² | 65K | 147K |
| w_g (门控) | 3d | 768 | 1.15K |
| **每层小计** | ~5d² | **328K** | **737K** |

- 12 层 d=256：3.9M + 词表嵌入(vocab 32K × 256 = 8.2M) = **12.1M**
- 12 层 d=384：8.8M + 词表嵌入(32K × 384 = 12.3M) = **21.1M**
- 16 层 d=384：11.8M + 12.3M = **24.1M**
- 16 层 d=512：15.7M + 16.4M = **32.1M**

均在 RTX 4080 16GB 可行范围内。

#### 在 10-50M 规模如何验证

1. **行为签名实验**：训练 PCN 和 Transformer（同参数量），对比：
   - 预测误差的空间分布（PCN 应该在"意外"位置产生高误差）
   - 跨位置门控模式（PCN 应该表现出误差聚类，而非内容聚类）
   - 层间 CKA（PCN 应该表现出不同的层级结构）

2. **recall vs 可预测性实验**：
   - 在可预测序列（周期性、n-gram）上，PCN 应该更高效
   - 在不可预测序列（随机 token）上，PCN 应该表现不同（可能更好或更差）
   - 关键：行为模式应该与 Transformer **定性不同**

3. **层级探针**：
   - 训练线性探针检测语法特征（词性、依赖关系、层级距离）
   - PCN 应该在更少的层就能捕获高阶语法特征（如果层级假设成立）

#### 最大风险

**跨位置交互可能退化为 attention**。如果门控函数 g_{ij} 学到的行为等价于 softmax 相似度，那 PCN 就只是"带反馈连接的 Transformer"。

### 候选 B：全息记忆网络（Holographic Memory Network, HMN）

#### 核心计算基元

**循环卷积绑定与相关解绑**（Circular Convolution Binding / Correlation Unbinding）

来源于向量符号架构（Vector Symbolic Architectures, VSA）和全息简化表示（Holographic Reduced Representations, HRR, Plate 1995）。

- Transformer 核心运算：矩阵乘法 `QK^T`
- SSM 核心运算：线性递归 `h_t = A h_{t-1} + B x_t`
- **HMN 核心运算**：循环卷积 `a ⊛ b = IFFT(FFT(a) ⊙ FFT(b))`

这是一个完全不同的代数结构——在频域中操作，信息通过绑定/解绑而非相似度匹配来存取。

#### 架构详述

```
HMN 层（Layer l），H 个记忆头
═══════════════════════════════════════════════

输入: X ∈ ℝ^{n×d}

步骤 1 — 投影到填充子和角色子:
  对每个头 h = 1...H:
    F_h = X · W_{F,h}        ∈ ℝ^{n×d_h}  (fillers)
    R_h = X · W_{R,h} + P · W_{P,h}  ∈ ℝ^{n×d_h}  (roles, 内容+位置)

步骤 2 — 绑定（循环卷积）:
  对每个位置 i:
    b_{h,i} = bind(F_{h,i}, R_{h,i})
           = IFFT(FFT(F_{h,i}) ⊙ FFT(R_{h,i}))   ∈ ℝ^{d_h}

步骤 3 — 叠加（构造全息记忆）:
  M_h = (1/n) Σ_{i=1}^{n} b_{h,i}   ∈ ℝ^{d_h}
  
  这是一个固定大小的记忆向量，编码了所有 token 的绑定信息。

步骤 4 — 查询与解绑（循环相关）:
  对每个位置 j，每个头 h:
    Q_{h,j} = X_j · W_{Q,h}    ∈ ℝ^{d_h}
    r_{h,j} = unbind(M_h, Q_{h,j})
           = IFFT(conj(FFT(Q_{h,j})) ⊙ FFT(M_h))  ∈ ℝ^{d_h}

步骤 5 — 清理（学习去噪）:
  c_{h,j} = W_{clean,h} · GELU(W_{denoise,h} · r_{h,j})   ∈ ℝ^{d_h}
  
  清理步骤是 HMN 区别于线性注意力的关键：
  它学习一个去噪自编码器来从含噪的全息检索中恢复干净信号。

步骤 6 — 合并与输出:
  Y_j = W_O · [c_{1,j}; c_{2,j}; ...; c_{H,j}] + X_j   ∈ ℝ^{d}

FFT 操作可批量并行，无需序列化。
```

#### 与线性注意力的关键差异

| 方面 | 线性注意力 | HMN |
|------|-----------|-----|
| 特征映射 | 随机/学习的 φ(·) | FFT（有明确的频域语义） |
| 叠加方式 | 外积 φ(k)⊗v 压缩到 d×d | 循环卷积保持在 d |
| 检索方式 | 内积 φ(q)^T · (φ(K)^T V) | 循环相关 IFFT(conj(FFT(q)) ⊙ FFT(M)) |
| 去噪 | 无 | 有（学习的清理网络） |
| 角色-填充子分离 | 无（k 和 v 是独立的） | 有（role 和 filler 的绑定是显式的） |
| 频域结构 | 无 | 有（不同频率组件独立携带不同信息） |

#### 为什么它在理论上可能学到不同的东西

1. **频域分解**：循环卷积在频域是逐频率的乘法，不同频率组件携带不同的绑定信息。这是 Transformer 不具备的多通道结构。
2. **角色-填充子结构**：HMN 显式分离"是什么"（filler）和"在什么角色上"（role），这更接近语言的结构（主语-动词-宾语等角色结构）。
3. **固定大小记忆**：不像 attention 需要存储所有历史，HMN 压缩到固定向量，但通过频域编码可能比 SSM 存储更多信息。
4. **误差纠正**：清理步骤学习从噪声中恢复信号，这赋予 HMN 一种"联想记忆"的能力，是 attention/SSM 都没有的。

#### 参数估算

| 组件 | 参数量 | d=256, H=4 | d=384, H=6 |
|------|--------|-----------|-----------|
| W_{F,h} (每头) | d × d_h | 16K × 4 = 64K | 24K × 6 = 144K |
| W_{R,h} (每头) | d × d_h | 64K | 144K |
| W_{P,h} (每头) | d × d_h | 64K | 144K |
| W_{Q,h} (每头) | d × d_h | 64K | 144K |
| W_{denoise,h} | d_h × d_h | 4K × 4 = 16K | 24K × 6 = 144K |
| W_{clean,h} | d_h × d_h | 16K | 144K |
| W_O | d × d | 65K | 147K |
| **每层小计** | | **~353K** | **~1.01M** |

- 12 层 d=256：4.2M + 8.2M (embed) = **12.4M**
- 12 层 d=384：12.1M + 12.3M = **24.4M**
- 16 层 d=384：16.2M + 12.3M = **28.5M**

#### 最大风险

**全息记忆的容量瓶颈**。HRR 的存储容量约为 O(d / log d) 个可靠恢复的项。对于 d_h=64，这意味着每个头只能可靠存储约 10-15 项。即使有多个头，总容量可能不足以处理长序列。这可能导致 HMN 在长序列上的 recall 任务严重退化——和 SSM 一样。

### 候选 C：稀疏激活脉冲网络（Sparse Activation Pulse Network, SAPN）

#### 核心计算基元

**脉冲编码 + 稀疏时间动力学**（Spike Coding with Sparse Temporal Dynamics）

受生物神经系统和脉冲神经网络（SNN）启发，但不是传统 SNN——使用连续松弛的脉冲机制实现端到端训练。

- Transformer：密集连续激活，每个位置每层都有非零输出
- SSM：密集连续状态，状态向量每个时间步都更新
- **SAPN：稀疏二值脉冲**，每个位置只激活一小部分神经元，信息通过稀疏脉冲的模式传递

#### 架构详述

```
SAPN 层（Layer l）
═══════════════════════════════════════════════

输入: X ∈ ℝ^{n×d}

步骤 1 — 膜电位累积:
  对每个位置 i:
    V_i = W_mem · x_i  + b_mem              ∈ ℝ^{d}
    （膜电位 = 线性投影 + 偏置）

步骤 2 — 稀疏脉冲生成:
  s_i = TopK(ReLU(V_i), k)                   ∈ {0, 1}^d
  （只有 k 个分量非零，模拟脉冲发放）
  
  实现方式（可微）:
  s_i = sigmoid(β · (V_i - θ_i)) · (V_i > θ_i)
  其中 θ_i 是学习阈值，β 是温度参数
  训练时用直通估计器（Straight-Through Estimator）

步骤 3 — 脉冲传播（跨位置）:
  对每个发放脉冲的神经元 j 在位置 i:
  通过学习的连接矩阵传播到其他位置:
  received_i = Σ_j W_{prop}[s_i, s_j] · pulse_value_j
  
  关键：传播矩阵 W_{prop} 是稀疏的（每个脉冲只连接到少数目标），
  且是内容依赖的（脉冲模式决定连接模式）

步骤 4 — 状态积分（时间累积）:
  h_i^{t} = λ · h_i^{t-1} + (1-λ) · (received_i + W_int · s_i)
  （积分-发放模型：累积输入直到阈值，然后发放）

步骤 5 — 输出:
  y_i = W_out · h_i^T  + x_i
```

#### 为什么它在理论上可能学到不同的东西

1. **信息瓶颈**：每个位置只有 k 个非零分量，强制模型学习极端压缩的表示。Transformer 的信息流是密集的。
2. **时间动力学**：积分-发放机制引入了真正的时间常数（λ），不同信息以不同速率积累和衰减。
3. **事件驱动**：只有"超过阈值"的信息才传播，这是一种基于**显著性**而非**相似度**的选择机制。
4. **能量效率**：稀疏激活意味着大部分计算可以跳过（理论上是 O(k) 而非 O(d)）。

#### 参数估算

| 组件 | 参数量 | d=256 | d=384 |
|------|--------|-------|-------|
| W_mem | d × d | 65K | 147K |
| θ (阈值) | d | 256 | 384 |
| W_{prop} | d × d (但稀疏，可减) | 65K → 16K(25% 稀疏) | 147K → 37K |
| W_int | d × d | 65K | 147K |
| W_out | d × d | 65K | 147K |
| **每层小计** | | **~210K** | **~478K** |

- 12 层 d=256：2.5M + 8.2M = **10.7M**
- 16 层 d=384：7.7M + 12.3M = **20.0M**

#### 最大风险

**直通估计器的梯度问题**。稀疏脉冲的不可微性导致梯度估计质量差。虽然 STE 在量化等领域可用，但在作为核心计算基元时可能导致训练不稳定或收敛到平凡解（如所有脉冲都相同）。

---

## 4. 推荐方向

### 推荐：候选 A — 预测编码网络（PCN）

#### 为什么选 PCN 而不是其他

| 判据 | PCN | HMN | SAPN |
|------|-----|-----|------|
| 与 Transformer 的根本差异 | ★★★ 双向信息流 | ★★ 不同代数结构但类比线性注意力 | ★★★ 完全不同的激活范式 |
| 理论动机强度 | ★★★ 神经科学+认知科学 | ★★ VSA 有数学基础 | ★★ SNN 有生物基础但语言建模联系弱 |
| 实现可行性 | ★★★ 标准矩阵运算 | ★★ 需 FFT，但有 CUDA 支持 | ★ 直通估计器训练困难 |
| 10-50M 可验证性 | ★★★ 行为签名清晰 | ★★★ 容量测试明确 | ★★ 稀疏模式分析复杂 |
| 最大风险的严重性 | 中（可能退化为 attention 变体） | 中（容量瓶颈） | 高（训练不稳定） |
| **与已知工作的重叠** | 低（语言建模中几乎未被探索） | 中（与线性注意力有关联） | 中（SNN 有相关工作） |

**核心理由：**

1. **预测编码从未在语言建模中被认真尝试过**。这是一个真正的空白——不是"做了一点但不够好"，而是"几乎没做过"。

2. **它回答了一个深刻的科学问题**：语言的层级结构是否更适合用层级预测来处理？这比"能否更高效地做序列建模"（HMN 的问题）或"能否用脉冲计算"（SAPN 的问题）更有意义。

3. **它有一个清晰的"非 Transformer"特征**：反馈连接。如果 PCN 的反馈连接是关键的（去掉后性能显著下降），那这就证明了非前馈架构的价值。

4. **风险可控**：即使 PCN 没有打败 Transformer，其行为签名（预测误差分布、层级结构、反馈 vs 前馈的贡献）也是有价值的发现。

#### 不选 HMN 的理由

HMN 虽然在代数结构上不同于 attention，但其在 Miras 框架下可以表述为"使用 FFT 特征映射 + 学习去噪的线性注意力"。这使得"HMN 不是 Transformer 变体"的论证困难。此外，容量瓶颈可能导致其在实际任务上严重退化，无法展示令人信服的结果。

#### 不选 SAPN 的理由

SAPN 的最大风险（训练不稳定）太严重。V1-V3 的教训之一是"训练超参可以主导架构变化"——如果 SAPN 需要极其精细的超参调节才能训练，那实验结果几乎不可解释。此外，直通估计器的梯度噪声可能导致"机制退化"（LESSONS.md 教训 3）更难诊断。

---

## 5. 实验设计

### 5.1 第一阶段：最小可区分实验（2-3 周）

#### 实验 1: 架构签名对比（1 周）

**目标**：确定 PCN 是否表现出与 Transformer 定性不同的行为。

**配置**：
- 模型：PCN-12L-d256 (~12M) vs Transformer-12L-d256 (~12M)
- 数据：TinyStories（适合小模型，故事级语言建模）
- 训练：5000 步，cosine schedule，相同学习率范围扫描

**测量**：
1. **PPL 曲线**：训练步数 vs validation PPL。关注形状差异（不仅最终值）
2. **预测误差的空间分布**（PCN 专属）：
   - 计算每层每位置的预测误差范数 ‖e_l,i‖
   - 可视化：是否在" surprising"token（低频词、句法边界）处产生高误差？
3. **门控模式分析**：
   - PCN 的 g_{ij} 是否与 Transformer 的 attention 权重 pattern 定性不同？
   - 测量：attention entropy vs PCN gating entropy
4. **层间 CKA（Centered Kernel Alignment）**：
   - 对比 PCN 和 Transformer 各层表示的 CKA 矩阵
   - 不同的 CKA 模式 = 不同的内部表示结构

**判定标准**：
- ✅ 至少 3 个测量维度上表现出定性差异
- ❌ 如果 PCN 的行为签名与 Transformer 几乎相同 → 说明反馈连接没有实质作用

#### 实验 2: 反馈消融（1 周）

**目标**：证明 PCN 的关键差异来自反馈连接（而非前馈部分的调参差异）。

**配置**：
- PCN-full：完整预测编码网络
- PCN-no-feedback：去掉 top-down 预测（p_l = 0），等价于前馈网络 + 误差驱动门控
- PCN-no-gating：去掉误差驱动门控，用标准 attention 替代（退化为带残差的 Transformer）
- Transformer：标准基线

**测量**：
1. PPL 对比
2. 各模型的表示相似性（CKA）
3. 在 recall 任务（MQAR simplified）上的表现

**判定标准**：
- ✅ PCN-full 与 PCN-no-feedback 有显著差异 → 反馈连接有实质作用
- ✅ PCN-full 的行为签名与 Transformer 有定性差异 → 不是 Transformer 变体
- ❌ 如果 PCN-no-feedback ≈ PCN-full → 反馈连接无用，PCN 本质上是前馈网络

#### 实验 3: 层级结构探针（1 周）

**目标**：验证 PCN 是否比 Transformer 更好地捕获层级结构。

**配置**：
- 训练 PCN-12L-d256 和 Transformer-12L-d256 在 TinyStories 上
- 对每层表示训练线性探针检测：
  - 词性标注（POS）
  - 依存关系（dependency depth）
  - 句子嵌套深度

**测量**：
- 每层探针准确率
- "语法信息涌现"的层数（达到 80% 准确率的最早层）

**判定标准**：
- ✅ PCN 在更少的层达到高准确率 → 层级预测促进语法学习
- ➖ PCN 和 Transformer 表现类似 → 层级优势不存在
- ❌ PCN 在所有层都显著差 → 架构有根本问题

### 5.2 第二阶段：规模验证（2-3 周）

如果第一阶段结果积极（至少 2/3 实验显示定性差异），进入第二阶段：

#### 实验 4: 扩大规模
- PCN-16L-d384 (~24M) vs Transformer-16L-d384 (~24M)
- 数据：OpenWebText-2 或 WikiText-103
- 训练：20K 步

#### 实验 5: 任务对比
- 在以下任务上对比：
  - 语言建模 PPL
  - MQAR（recall 任务）
  - 合成语法任务（主谓一致、反射代词）
  - 复制任务（随机 token 序列复制）

#### 实验 6: 计算效率对比
- 训练/推理速度
- 内存占用
- 与 attention 的实际计算量对比

### 5.3 资源估算

| 阶段 | GPU 时间 | 显存需求 | 日历时间 |
|------|---------|---------|---------|
| 第一阶段 | ~40 GPU·小时 | ~6 GB (d=256, batch=32) | 2-3 周 |
| 第二阶段 | ~80 GPU·小时 | ~12 GB (d=384, batch=16) | 2-3 周 |
| 总计 | ~120 GPU·小时 | RTX 4080 可行 | 4-6 周 |

---

## 6. 风险评估

### 6.1 最大失败可能性

**风险 1（高）：PCN 退化为带额外组件的 Transformer**

描述：训练过程中，PCN 的门控函数 g_{ij} 可能学到等价于 softmax 相似度的行为，反馈连接可能学到接近零的权重，最终模型的行为与 Transformer 不可区分。

概率：40-50%

V1-V3 的教训明确表明："精心设计的机制在训练中自动退化为更简单的版本"是一个常见模式。如果反馈信号提供了很小的收益，优化器可能直接忽略它。

**缓解措施**：
- 实验 2 的消融实验会直接检测这一退化
- 可以在训练中添加反馈连接的 L1 正则化（强制保留反馈信号）
- 如果 g_{ij} 退化为 softmax，这是一个发现本身——说明 attention 是预测误差门控的吸引子

**风险 2（中）：PCN 的反馈连接导致训练不稳定**

描述：双向信息流可能导致梯度爆炸/消失，或者训练振荡不收敛。

概率：20-30%

**缓解措施**：
- 使用梯度裁剪和 LayerScale（类似 CaiT）
- 反馈连接初始化为接近零，逐渐增加
- 如果不稳定，可以改为"前馈训练 + 反馈推理"模式

**风险 3（中）：TinyStories 上的结果不可推广**

描述：LESSONS.md 教训 1 指出"12 层 d=256 的实验结果不能推广"。TinyStories 的简单性可能使得 PCN 和 Transformer 的差异被正则化效应掩盖。

概率：15-25%

**缓解措施**：
- 第二阶段扩展到 OpenWebText / WikiText
- 设计对抗性实验：构造 Transformer 容易但 PCN 应该难的数据（反之亦然）
- 关注定性差异（行为签名），而非定量 PPL

### 6.2 如果失败了，退路

**情景 A：PCN 退化为 Transformer**

退路：分析退化过程本身。记录 g_{ij} 的训练动态——它如何从"误差驱动"演变为"相似度驱动"？这个分析本身就是一个有价值的贡献，揭示了 attention 作为优化吸引子的力量。

**情景 B：PCN 训练不稳定**

退路 1：简化为"单向前馈 + 误差门控"（去掉反馈连接），测试这个简化版本是否仍有独特行为。如果有，这就是一个"误差驱动的注意力替代"——虽然不如完整 PCN 有雄心，但仍有价值。

退路 2：转向 HMN 方向。如果 PCN 不可行，HMN 有明确的容量测试和频域分析工具，更容易诊断问题。

**情景 C：PCN 可以训练，但在所有指标上与 Transformer 不可区分**

退路：这是一个强有力的否定结论——"预测编码原理在当前规模下不能产生不同于 Transformer 的语言模型"。结合 V1-V3 的结论，这进一步缩小了可能的设计空间，对后续工作有指导意义。

### 6.3 什么条件下能说"V4 证明了一个新架构方向"？

需要**同时满足**：

1. **PCN 在至少 2 个行为维度上与 Transformer 有定性差异**（非调参可解释）
2. **反馈消融实验证明反馈连接是关键**（去掉后行为改变）
3. **在至少一个任务上，PCN 展示出 Transformer 不具备的能力模式**（不需要"更好"，需要"不同"）
4. **结果在 d=384 规模上可复现**（不只是 d=256 的小规模伪影）

如果这四条都满足，V4 的贡献是：**证明了非前馈、预测误差驱动的架构在语言建模中具有独特价值，开辟了预测编码语言模型的研究方向。**

### 6.4 诚实的可行性评估

**PCN 能在 PPL 上打败 Transformer 吗？大概率不能。**

Transformer 经过多年的工程优化（初始化、归一化、学习率调度等），在新架构的工程经验为零的条件下，PPL 很可能劣于 Transformer。

**但 PPL 不是 V4 的目标。** V4 的目标是证明"存在不同于 Transformer 的有效语言建模架构"。如果 PCN 展示出独特的表示结构、不同的学习曲线、以及对反馈连接的依赖，这就够了。

**真正的风险不是"不够好"，而是"不够不同"。** 如果 PCN 最终只是一个"带反馈的 Transformer"，那它就不满足成功标准。

---

## 附录 A：PCN 实现要点

### A.1 PyTorch 伪代码

```python
class PCNLayer(nn.Module):
    """预测编码层"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d = d_model
        
        # 底层投影
        self.W_bu = nn.Linear(d_model, d_model, bias=False)
        self.W_td = nn.Linear(d_model, d_model, bias=False)  # top-down
        self.W_lat = nn.Linear(d_model, d_model, bias=False)  # lateral
        self.W_up = nn.Linear(d_model, d_model, bias=False)   # update
        self.W_res = nn.Linear(d_model, d_model, bias=False)  # residual
        
        # 误差门控参数
        self.w_g = nn.Linear(3 * d_model, 1, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, h_above=None):
        """
        x:         [batch, seq, d] — 来自下层的输入
        h_above:   [batch, seq, d] — 来自上层的状态（首层为 None）
        """
        # 步骤 1: 自底向上特征
        u = self.norm(self.W_bu(x))  # [B, N, D]
        
        # 步骤 2: 自顶向下预测
        if h_above is not None:
            p = self.W_td(h_above)   # [B, N, D]
        else:
            p = torch.zeros_like(u)
        
        # 步骤 3: 预测误差
        e = u - p  # [B, N, D]
        
        # 步骤 4: 误差驱动的跨位置交互
        B, N, D = e.shape
        e_i = e.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        e_j = e.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        
        # 门控值：基于误差交互
        gate_input = torch.cat([
            e_i * e_j,         # 误差交互
            e_i + e_j,         # 误差叠加
        ], dim=-1)              # [B, N, N, 2D]
        # 简化版门控（计算量控制）
        # 更高效的实现可用 linear + relu + linear
        g = torch.sigmoid(self.w_g(gate_input).squeeze(-1))  # [B, N, N]
        
        # 稀疏化门控（Top-K）以控制计算量
        # k = min(64, N)  # 每个位置最多关注 64 个
        # topk_vals, topk_idx = g.topk(k, dim=-1)
        # g_sparse = torch.zeros_like(g).scatter_(-1, topk_idx, topk_vals)
        
        # 加权误差聚合
        lat = torch.bmm(g, self.W_lat(e))  # [B, N, D]
        
        # 步骤 5: 状态更新
        h = F.gelu(self.W_up(e + lat)) + self.W_res(x)
        
        return h, e  # 返回状态和误差（用于分析）
```

### A.2 计算复杂度分析

| 步骤 | 计算量 | 内存 |
|------|--------|------|
| 底层投影 | O(N · d²) | O(N · d) |
| 顶向下预测 | O(N · d²) | O(N · d) |
| 误差计算 | O(N · d) | O(N · d) |
| 门控（全量） | O(N² · d) | O(N²) |
| 门控（Top-K, K=64） | O(N · K · d) | O(N · K) |
| 状态更新 | O(N · d²) | O(N · d) |
| **总计（Top-K）** | **O(N · (K + d) · d)** | **O(N · (K + d))** |

与 Transformer 对比：Transformer 注意力为 O(N² · d)，PCN 用 Top-K 门控为 O(N · K · d)，K=64 时远小于 N²。

### A.3 训练配置建议

```python
# 第一阶段配置
config = {
    "model": {
        "type": "PCN",
        "n_layers": 12,
        "d_model": 256,
        "dropout": 0.1,
        "topk_gating": 64,  # 每位置最多关注 64 个
    },
    "training": {
        "dataset": "TinyStories",
        "batch_size": 32,
        "seq_len": 512,
        "lr": 3e-4,
        "lr_schedule": "cosine",
        "warmup_steps": 500,
        "total_steps": 5000,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    },
    "logging": {
        "log_interval": 50,
        "save_error_patterns": True,   # 保存预测误差模式
        "save_gate_patterns": True,    # 保存门控模式
        "compute_cka": True,           # 计算层间 CKA
    }
}
```

---

## 附录 B：完整文献列表

| # | 论文 | arXiv | 关键贡献 | 与 V4 的关系 |
|---|------|-------|---------|-------------|
| 1 | Mamba | 2312.00752 | 选择性 SSM | SSM 的上限，recall 局限 |
| 2 | Mamba-2 | 2405.21060 | SSM-attention 数学等价 | 证明 SSM 不是"新范式" |
| 3 | RWKV | 2305.13048 | 线性注意力 RNN | 线性注意力变体 |
| 4 | Hyena | 2302.10866 | 长卷积 + 门控 | 卷积范式的代表 |
| 5 | RetNet | 2307.08621 | 保持机制 | 统一 attention 和递归 |
| 6 | Griffin | 2402.19427 | 门控递归 + 局部 attention | 混合架构仍需 attention |
| 7 | BASED | 2402.18668 | 回忆-效率权衡 | 识别了核心 tradeoff |
| 8 | Repeat After Me | 2402.01032 | SSM 不能复制 | 固定状态的信息论限制 |
| 9 | Titans | 2501.00663 | 神经长期记忆 | 引入 test-time learning |
| 10 | Miras | 2504.13173 | 统一框架 | 证明所有模型 = 联想记忆 + 偏置 |
| 11 | GoldFinch | 2407.12077 | 混合 + 压缩 cache | 工程优化 |
| 12 | DyT | 2503.10622 | tanh 替代归一化 | 训练技巧 |

---

## 附录 C：实验代码结构

```
prism-v4/
├── configs/
│   ├── pcn_tiny.yaml          # PCN-12L-d256
│   ├── transformer_tiny.yaml  # Transformer-12L-d256 (对照)
│   ├── pcn_medium.yaml        # PCN-16L-d384
│   └── transformer_medium.yaml
├── models/
│   ├── pcn.py                 # PCN 架构实现
│   ├── pcn_layer.py           # PCN 单层实现
│   ├── transformer.py         # 标准 Transformer（对照）
│   ├── hmn.py                 # HMN（候选 B，备用）
│   └──sapn.py                 # SAPN（候选 C，备用）
├── experiments/
│   ├── exp1_signatures.py     # 架构签名对比
│   ├── exp2_ablation.py       # 反馈消融
│   ├── exp3_probing.py        # 层级探针
│   ├── exp4_scaling.py        # 规模验证
│   └── exp5_tasks.py          # 任务对比
├── analysis/
│   ├── cka.py                 # CKA 计算
│   ├── error_visualization.py # 预测误差可视化
│   ├── gate_analysis.py       # 门控模式分析
│   └── capacity_test.py       # HMN 容量测试
├── data/
│   └── prepare.py             # 数据预处理
└── train.py                   # 统一训练脚本
```

---

*2026-06-24 | PRISM V4 方案设计 | 基于 V1-V3 教训 + 2023-2026 文献调研*
