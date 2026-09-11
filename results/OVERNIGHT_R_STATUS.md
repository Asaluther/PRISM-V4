# 夜间 R 队列执行报告

- 完成时间: 2026-09-09 02:27:20
- 执行策略: 单项失败不中断；全部完成后关机（用户授权）

## 1. d=384 规模复现（TinyStories；d256 对照: TF 15.2 / NG 16.5 / 2P 20.1）

| 配置 | PPL | tokens/s |
|---|---|---|
| o384_tf | 21.65 | 87414 |
| o384_ng | 12.39 | 86693 |
| o384_2p | 15.75 | 17446 |

## 2. 公平 lr 的 20K 走势（WT；5K 对照: TF 214.5 / NG 176.0）

- TF@1e-3 20K 步: 53.1

## 3. lr 5e-4 细扫（WT；对照: NG@3e-4=176.0 / 2P@1e-3=201.5）

- wt_ng_lr5e4_s0: 147.78
- wt_2p_lr5e4_s0: 200.90

## 4. triton-windows / compile 基准

- transformer: 181,005 tok/s
- pcn_eager: 28,399 tok/s
- pcn_compile: 18,678 tok/s

## 5. 机制实验多 seed（7 变体 × 5 seed，学会判据 copy<1.0）

| 变体 | n | copy 均值 | 学会率 |
|---|---|---|---|
| tf_softmax(原版) | 5 | 3.435 | 0/5 |
| tf_sigmoid(门控式) | 5 | 3.434 | 0/5 |
| tf_sm_t0.5 | 5 | 3.435 | 0/5 |
| tf_sm_t2.0 | 5 | 3.435 | 0/5 |
| pcn_sigmoid(原版) | 5 | 0.233 | 5/5 |
| pcn_softmax(2c) | 5 | 1.109 | 3/5 |

## 6. 门来源 × 值通路 2×2 分解

- 未产出（见 overnight_r.log）

---
完整日志: results/overnight_r.log | 系统即将关机（用户已授权）
