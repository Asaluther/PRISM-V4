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

---

## 勘误（2026-09-11 晚，公平基线复核 + 遗忘基准后）

两个口径修正，详见 `../mechanism/SKELETON_CONSTRAINT.md` 勘误节与 `../mechanism/FORGETTING_BENCHMARK.md`：

1. **TF 基线次优**：本 demo 的 TF 臂用 `wt_causal_transformer_s0`（lr 3e-4，WT PPL 557）。
   公平版 TF（`wt_tf_lr1e3_s1`，PPL 215）同协议微调**改善 +5.4%** 而非 -19.7%——
   「TF 过拟合退化」是次优基线伪影（教训 8 复发）。修正后对比：**PCN +58.5% vs TF +5.4%**，
   PCN 跨域适应优势依然大幅成立（且起点 PPL 更差），但定性叙事从「TF 变差」改为「TF 几乎不动」。
2. **「个性化」重定性为「跨域适应」**：域内（TS base→TS 用户数据）24 格校准网格双架构
   无一为正——1.5K token 同域数据在 22M 收敛模型上无可挖增益。本 demo 的增益全部来自
   WT→TS 域差闭合，与二维规律 +84% 低数据象限一致。终端叙事应表述为
   「换域/冷启动快速适应」而非「同域风格个性化」。
