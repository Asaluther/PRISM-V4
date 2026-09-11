# PRISM V4 — 行为签名分析（证据线 A | ckpt=causal_ng_wt）

- 误差-惊讶度 Pearson 相关（PCN, 12 层平均）: **0.015**
  （预测编码假说预测正相关；0 附近=误差与惊讶度无关）
- PCN 门控熵（归一化分布, nats）: N/A（no_gating 消融无门控） / 均匀上限 ln256=5.545
- Transformer attention 熵: N/A
- 跨模型 CKA 均值: 0.469（低=两模型表示结构不同）
- PCN 内部 CKA 对角均值: 1.000
- TF 内部 CKA 对角均值: 1.000

## 训练曲线
