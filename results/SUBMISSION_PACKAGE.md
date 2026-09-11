# ChinaXiv 投稿材料包

> 生成于 2026-09-10 | 论文主体：PRISM_V5_TECH_REPORT.md
> 个人信息留空由作者填写；实际提交在 ChinaXiv 平台完成

---

## 一、作者信息（填空）

| 项 | 内容 |
|---|---|
| 中文姓名 |邓克瑜 |
| 英文/拼音姓名 | Dengkeyu |
| 单位 | 独立研究者 / Independent Researcher |
| 电子邮箱 | asaluther@live.com |
| ORCID | 0009-0003-6991-6148 |
| 通讯地址 | ＿＿＿＿＿＿（平台要求时填写） |

## 二、中文元数据

**标题**：
预测编码网络在语言建模中的样本效率、机制不可达性与测量纪律：一项单卡架构研究

**摘要 v2**（规范版，约 300 字）：
我们报告预测编码网络（Predictive Coding Network, PCN）在语言建模上的一项系统性单卡研究，得到四个主要发现。第一，**二维不对称规律**：误差流变体相对 Transformer 的优势沿两条正交轴呈不同形态——减数据轴（固定模型、数据量递减）优势单调上升至 +84%；加参数轴（固定数据、参数递增）呈倒 U 形，峰值出现在约 1000 tokens/parameter（WikiText-103 上 +60.8%）。该规律对终端个性化场景（冻结中等模型+少量用户数据微调）具有直接部署指导意义。第二，组合性质与对称不可达性：PCN 在小样本精确复制任务上的能力（15/15 对同规模 Transformer 的 0/30）经 13 个变体双向对照和 2×2×2 全因子设计（8 格全部成功），被证明不可分解为任何单一组件——从标准 Transformer 块移植组件无法获得该能力。第三，局部学习规则的适用边界：Whittington-Bogacz 式局部规则在多层感知机上与反向传播完全等价，但在 PCN 结构上存在 2.4-3.7 倍的结构性差距。第四，测量纪律：四次系统性测量偏差（非因果泄露、基线超参次优、自建对照失效、评估双重移位）均被预设验证程序捕获，我们将其提炼为可复用的方法学条款。结果表明 PCN 是数据稀缺区间的优势架构；其能力是不可移植的组合性质。

**关键词**：
非 Transformer 架构；预测编码；样本效率；消融方法学；因果性；语言模型；独立研究

**学科分类建议**：
计算机科学 → 人工智能 / 机器学习（主）；计算语言学（辅）

**数据与代码可用性声明**：
本研究的全部实验数据（约 130 个训练运行的结果文件）、分析脚本与可复现管线已在配套代码库中完整保留；论文中全部数字可由 results/ 目录下的 JSON 结果文件与 run_*.sh 幂等脚本逐一核验。代码库公开计划：〔选择：随论文公开 / 按请求提供 / 后续公开〕。

## 三、英文元数据（arXiv 第二步备用）

**Title**:
Sample Efficiency, Mechanistic Unreachability, and Measurement Discipline of Predictive Coding Networks in Language Modeling: A Single-GPU Architecture Study

**Abstract**:
We report a systematic single-GPU study of Predictive Coding Networks (PCN) on language modeling, yielding four main findings. First, a scale-to-data ratio law: the advantage of the error-stream variant over Transformers follows an inverted-U shape as tokens-per-parameter decreases, peaking near 1000 (+60.8% on WikiText-103) and reversing under data abundance; the law successfully predicted one held-out data point. Second, an emergent compositional property with symmetric unreachability: PCN's capacity on small-budget exact copying (15/15 vs 0/30 for same-scale Transformers) survives component removal but cannot be transplanted into standard blocks across 13 bidirectionally-controlled variants. Third, the applicability boundary of local learning rules: Whittington-Bogacz-style rules exactly match backpropagation on MLPs yet exhibit a structural 2.4-3.7x gap on PCN, with learning-rate and iterative-inference confounds symmetrically ruled out. Fourth, measurement discipline: three systematic measurement biases (non-causal leakage, suboptimal baseline hyperparameters, invalid custom controls) were each captured by pre-registered verification procedures, which we distill into reusable methodological clauses. Together, these position PCN as an architecture advantageous in the low-data-per-parameter regime—precisely the terminal personalization scenario—with capabilities that emerge only from the non-Transformer backbone as a whole.

**Keywords**:
non-Transformer architectures; predictive coding; sample efficiency; ablation methodology; causality; language models

**arXiv 分类建议**：cs.LG (Machine Learning) 主；cs.CL 辅

## 四、提交流程 Checklist

### ChinaXiv（第一步）
1. [ ] 官网 chinaxiv.org 注册个人账号（实名认证，按页面指引）
2. [ ] 注册 ORCID（orcid.org，备用）
3. [ ] 上传论文 PDF（PRISM_V5_TECH_REPORT.md 转 PDF——需要转换时告诉我）
4. [ ] 填写元数据（用本文件第二节的中文版）
5. [ ] 选择学科分类 + 填写可用性声明
6. [ ] 提交 → 等待平台审核（周期通常数天，以平台通知为准）
7. [ ] 通过后记录 DOI 与时间戳（优先权确立）

### arXiv（第二步，ChinaXiv 通过后）
1. [ ] ⚠️ 2026 新规：需先获得 cs.LG 域的背书（endorsement）——
   途径：通过学术社区联系有背书资格的研究者（报告引用文献的作者、社区求助等）
   参考：info.arxiv.org/help/endorsement.html
2. [ ] 注册 arXiv 账号（真实姓名 + Independent Researcher）
3. [ ] 用第三节英文元数据提交（论文需英文版——需要时我翻译全文）

### 通用注意
- 论文内容与预印本平台政策：CC 许可协议不可撤销，发布前内容定稿
- 本报告为纯科学发现（无可专利技术方案内容），无「先专利后发表」时序约束
  （已确认 2026-09-10）；未来端侧方法等应用性成果需先专利后公开

---

*材料包完。作者信息与「公开计划」选项由作者定夺后填入。*

## 五、v2 版本变更说明（供 AiraXiv 版本备注）

| 变更 | 内容 |
|---|---|
| 核心修正 | 「倒 U 规律」→「二维不对称规律」：减数据轴单调至 +84%（v1 未测的左端），加参数轴倒 U 峰值 +60.8% |
| 统计强化 | 全部 11 数据点补 bootstrap 95% CI；「全胜」稳健性声明（NG 最差 seed 优于 TF 最好 seed） |
| 因子设计 | 补齐 2×2×2 全因子矩阵 8 格（响应审稿建议），组合性质升级为完整因子空间确认 |
| 效率反转 | 门控 bmm 数学重构：训练吞吐从 TF 的 16% 提升至 96%，推理 batch≥16 反超 TF +14% |
| 超参敏感性 | warmup 对照揭示 ~10% 效应远小于低数据区优势（59-84%），主结论免疫 |
| 方法论 | 术语表（附录 C）、审稿回应索引（附录 D）、第四次探测器实战记录 |

---

*材料包 v2 完。*
