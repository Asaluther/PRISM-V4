# V6 端侧 Demo：个人风格助手（2026-09-11）

> 终端愿景第一个可见里程碑——PCN 在 1.5K token 用户数据上 4.6 秒完成个性化适应

## 核心数据

| 指标 | PCN（误差流） | Transformer |
|---|---|---|
| 平均 PPL 改善 | **+58.5%** | **−19.7%（过拟合退化）** |
| 胜出用户 | **5/5 全胜** | 0/5 |
| 微调耗时 | 4.6 秒 | 3.4 秒 |
| 用户数据 | 1,536 tokens（6 段话） | 同 |
| 微调策略 | 冻结前 6 层 + embedding，只调后半 + head | 同 |

## 关键发现

**Transformer 在少数据微调后 PPL 反而恶化**（平均 −19.7%）——过拟合了 1536 token 的训练块，在 held-out 上变差。PCN 同数据同预算**改善 58.5%**——学到了泛化的风格特征而非死记训练块。

这直接验证了二维规律 +84% 象限在终端场景的表现：
- **PCN：少数据 → 快适应 → 泛化**
- **TF：少数据 → 过拟合 → 退化**

## 生成对比（定性）

预训练于 WikiText（百科），用户数据来自 TinyStories（儿童故事）——跨域个性化。

| | 微调前 | 微调后 |
|---|---|---|
| PCN | "episode", "= = History = =", "series" | "glass room she could", "really feel" |
| TF | "German", "000", "States" | 仍有 "situation episode decided" |

PCN 生成方向性转向故事叙事风格；TF 保留更多百科残留。

## 终端部署指标

| 指标 | 值 | 终端可行性 |
|---|---|---|
| 适应耗时 | 4.6 秒 | ✅ <5 秒 |
| 可训练参数 | ~10M（半模型） | ✅ 显存友好 |
| 用户输入 | 1.5K token（几段话） | ✅ 真实可用 |
| 硬件 | RTX 4080（终端 GPU） | ✅ |

## 文件

- `personalization_demo.py` — 定量对照脚本
- `generation_demo.py` — 生成对比脚本
- `personalization_demo.json` — 5 用户 PPL 数据
- `generation_comparison.json` — 生成文本
