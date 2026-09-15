# C4 — 私有 AI 安全框架文档

> 2026-09-16 | 配套代码：c_security_framework.py | 测试：results/v7/c_security_tests.json

## 威胁模型

| # | 威胁 | 场景 | 防护层 |
|---|---|---|---|
| T1 | **投毒适应** | 恶意输入导致模型学到有害模式 | C1 输入过滤 + 速率限制 |
| T2 | **过度适应退化** | 频繁微调导致模型崩溃 | C1b 速率限制（频率 + 总步数双控） |
| T3 | **适应不可逆** | 用户无法撤销 AI 的学习 | C2 checkpoint 快照 + 一键回滚 |
| T4 | **模型被盗** | 模型文件被复制到他人设备 | C3 所有者绑定（密码派生身份） |
| T5 | **适应层数据泄露** | 适应层的权重泄露用户偏好 | C3 选择性权重加密 |
| T6 | **数据出端** | 训练/适应数据被上传 | B4 已验证全本地（无网络调用） |

## C1 输入安全过滤

| 规则 | 参数 | 防护目标 |
|---|---|---|
| 最小 token 数 | 50 | 防止无意义适应 |
| 最大 token 数 | 2048 | 防止内存/时间攻击 |
| 重复率上限 | 70% | 防止循环垃圾输入 |
| 字符多样性 | ≥10 unique chars | 防止无意义字符流 |
| 注入模式检测 | 5 种模式 | 防止指令注入 |

## C1b 适应速率限制

| 参数 | 默认值 | 防护 |
|---|---|---|
| 每小时最大适应次数 | 5 | 防止频繁微调退化 |
| 总步数上限 | 5000 | 防止累计过拟合 |

## C2 可审计性

- **每次适应前**：自动保存模型 checkpoint
- **适应日志**：时间戳 + 输入 hash + token 数 + 步数 + 耗时
- **回滚机制**：用户可回滚到任意历史 checkpoint
- **审计输出**：JSON 文件（用户可直接阅读）

## C3 所有者绑定

- **身份派生**：PBKDF2-SHA256（100K 迭代）从密码派生唯一 owner_id
- **模型标识**：owner_id 嵌入模型元数据
- **加载验证**：加载时检查 owner_id 与当前密码是否匹配
- **选择性加密**：适应层权重用密钥流加密（W_up/W_res），基础层保持明文（性能）

## 局限声明

1. **非对抗级安全**：适合个人使用场景，不适合高安全需求（军事/金融级）
2. **加密为轻量级**：XOR 流密码，防偷窥不防专业破解
3. **注入检测为模式匹配**：简单规则，可被精心构造的输入绕过
4. **所有者绑定为软件级**：无硬件 TPM/Secure Enclave 支持
5. **未覆盖**：模型侧信道攻击、物理访问攻击、供应链攻击

## 与终端部署的集成

```python
# 完整安全流程
owner = OwnerBinding(password)           # C3: 身份
filter = InputFilter()                   # C1: 过滤
limiter = AdaptationRateLimiter()        # C1b: 限速
auditor = AdaptationAuditor(model, dir)  # C2: 审计

# 适应前检查
ok, issues = filter.check(tokens, text)
ok2, reason = limiter.can_adapt(steps)
if ok and ok2:
    auditor.snapshot_before(hash, n_tok)  # C2: 快照
    model = finetune(model, data)        # 适应
    limiter.record(steps)                # 记录
```
