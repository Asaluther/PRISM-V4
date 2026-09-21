# V7 终端形态第一里程碑：端到端私有 AI 演示

> 日期：2026-09-21 | 平台：本地纯 CPU（8 线程，零网络调用）
> 脚本：`v7_demo.py`（组装脚本）+ `int4_gate.py`（前置门槛）
> 模型：`wt_096m_hyb_h2e-4_s0`（h2e4 终端默认，96M，12TF+12PCN）
> 数据：`results/v7/{int4_gate,v7_demo_fp32,v7_demo_int4}.json`

## 一、里程碑含义

北极星（终端私有 AGI：个人设备可运行、低功耗、可在线学习、数据不出端）的
**第一个可见里程碑**落地：此前各零件（混合检查点 / 冻结骨干适应 / 50 步预算 /
int4 配方 / 安全框架 / CPU 链路）分别验证过，本里程碑把它们**串联为一个可
运行的端到端系统**——`PrivateAI` 类 + 脚本化演示 + 交互 REPL。

## 二、判据记分板（预注册，fp32 与 int4 双模式均 5/5）

| 判据 | fp32 | int4 |
|---|---|---|
| ① 端到端纯 CPU 通过（含所有者绑定双向验证） | ✅ | ✅ |
| ② 50 步用户适应 ≤30s 且改善为正 | ✅ 19.4s，**+70.1%** | ✅ 18.8s，+59.8% |
| ③ int4 损失 ≤1%（门槛实验） | —（引用） | ✅ **+0.15%** |
| ④ 生成可用（本地 tokenizer，采样） | ✅ 38.0 tok/s | ✅ 35.6 tok/s |
| ⑤ 回滚可演示（PPL 精确恢复） | ✅ | ✅ |

## 三、系统数字总览（纯 CPU）

| 指标 | fp32（366MB） | int4 fake-quant（100MB） |
|---|---|---|
| 模型加载 | 0.9s | 同级 |
| 推理吞吐（b1×256） | 1259 tok/s | 1528 tok/s |
| 采样生成 | 38.0 tok/s（26.3 ms/tok） | 35.6 tok/s（28.1 ms/tok） |
| 用户适应（50 步，冻结 TF 骨干） | 19.4s → held-out +70.1% | 18.8s → +59.8% |
| 快照回滚 | 精确恢复 | 精确恢复 |

注：int4 为 RTN fake-quant（数值验证 + 解析体积 100MB），**packed int4 推理
kernel 未实现**——端到端速度仍在 fp32 tensor 上测得；真实部署体积收益需要
kernel 化（后续工程项）。

## 四、发现与诚实边界

1. **int4 配方平移到混合架构近无损**（+0.15% @ 96M，T6 的 21M 结论外推成功；
   W_res 豁免 + embedding 豁免在 HybridModel 上语义不变）——顺带回应了评审
   Q4 的量化边界问题
2. **量化权重上直接适应会损失 ~10pp 增益**（+59.8% vs +70.1%）：50 步适应
   发生在 fake-quantized 的 PCN 头权重上。更优部署形态是「int4 冻结底座 +
   小块 fp 适应头」——列入后续工程（与 PEFT-on-hybrid 的 LoRA 路线衔接）
3. 生成样本为 WikiText 风格（h2e4 是 WT 预训练检查点，域 sanity 正确）；
   用户适应的 held-out 改善（+70%）与 phase3/step50 全线数字同型
4. 安全四件套全部接入并演示：输入过滤（注入/重复/长度）、速率限制
   （5 次/小时 + 5000 步总量）、快照回滚（审计轨迹）、PBKDF2 所有者绑定
   （错误口令拒绝验证）

## 五、使用方式

```
.venv/Scripts/python.exe v7_demo.py                # 脚本化演示（出 JSON + 记分板）
.venv/Scripts/python.exe v7_demo.py --quant int4   # int4 模式
.venv/Scripts/python.exe v7_demo.py --interactive  # REPL：/adapt /gen /ppl /rollback
```

## 六、后续（V7 路线图剩余）

- packed int4 kernel 化（真实体积/能耗收益）
- int4 底座 + fp 适应头的两段式部署形态
- 真实用户数据试点 + 长时程流（与论文侧挂账项同源）
- 能耗口径补测（v1 的 T4 方法平移到 96M 混合）

## 七、packed int4 实测补记（2026-09-21 下午，int4_kernel.py）

把 fake-quant 推进到**真实打包存储**（int4-g64 非对称，两权重/字节 uint8，
路线 B 自写实现——torchao 路线 A 在 Windows 上被 mslk 内核库缺失阻塞）：

| 指标 | fp32 | packed int4 | 判据 |
|---|---|---|---|
| 权重驻留（tensor 存储） | 366 MB | **201 MB**（Linear 部分仅 ~95MB；emb 103MB fp32 + MHA 未打包） | ≤110MB ❌ |
| 进程 RSS | 1338 MB | 1324 MB（-1%——逐前向反量化缓冲抵消收益） | -30% ❌ |
| 生成 | 37.5 tok/s | 13.6 tok/s（无融合 kernel，反量化 2.7× 开销） | ≥25 ❌ |
| PPL 损失 | — | **+0.39%** | ≤1% ✅ |

**诚实结论**：打包与数值在 Windows CPU 上可行且近无损，但**没有融合
dequant-GEMM kernel 就没有终端价值**（RSS 不降、速度倒退）。部署级 int4 的
可行路径二选一：(a) torchao+torch.compile（需 Linux 或 mslk 可用环境）；
(b) GGUF/llama.cpp 移植（工程量 ~1 周，列为独立后续项）。V7 demo 的 int4
模式维持 fake-quant 口径（数值验证）+ 本节的真打包实测记录。
