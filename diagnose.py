#!/usr/bin/env python3
"""PRISM V4 — 第 1 步：零训练诊断

解剖 results/pcn_v1/best_model.pt（val PPL 406, loss 卡 6.0），不训练。
回答的问题：PCN 为什么五千步不学习？

嫌疑清单（来自代码审查，本脚本逐一实测）：
  S1  W_res 非恒等残差 + std=0.02 初始化 → 谱范数 ~0.6 → 信号/梯度穿 12 层指数衰减
      （Transformer 残差是恒等映射 x + f(x)，无此问题）
  S2  门控 w_g 初始化后 g ≈ sigmoid(0) = 0.5，训练中若 w_g 梯度消失则永远均匀
  S3  误差 e 的 clamp(-10,10) 饱和
  S4  深层梯度消失：靠近输入的层梯度范数 << 靠近输出的层
  S5  上下文失明：模型输出只依赖 unigram 分布，不使用上下文
      （loss(原序) ≈ loss(位置打乱) 即实锤）

输出：results/diagnosis/step1_diagnosis.json + 控制台报告
"""

import sys
import json
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

VOCAB = 50257
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('results/diagnosis')


def make_pcn():
    # 与 results/pcn_v1/results.json config 完全一致（含 max_seq_len=128, pos_emb 尺寸）
    return PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                    d_gate=32, dropout=0.1, max_seq_len=128)


def random_batch(B, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, N), generator=g)


def lm_loss(logits, ids):
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))


@torch.no_grad()
def recompute_errors(model, layer_inputs):
    """用抓到的每层输入复算预测误差 e_l 与门控 g（与 PCNLayer.forward 同公式）。

    第 i 层: e_i = W_bu(LN(x_i)) - W_td(x_{i-1})，第 0 层 p=0
    """
    stats = []
    for i, layer in enumerate(model.layers):
        pc = layer.pcn_layer
        x_i = layer_inputs[i]
        u = pc.W_bu(pc.norm_bu(x_i))
        if i > 0:
            p = pc.W_td(layer_inputs[i - 1])
            e = u - p
        else:
            e = u
        # clamp 与 forward 一致
        e_clamped = e.clamp(-10.0, 10.0)
        sat = (e.abs() > 9.9).float().mean().item()

        # 门控复算：e_comp -> g = sigmoid(w_g([e_i*e_j, e_i+e_j]))
        e_comp = pc.W_compress(e_clamped)
        B, N, Dg = e_comp.shape
        e_i = e_comp.unsqueeze(2)
        e_j = e_comp.unsqueeze(1)
        inter = torch.cat([e_i * e_j, e_i + e_j], dim=-1)
        g = torch.sigmoid(pc.w_g(inter).squeeze(-1))  # [B, N, N]

        # Top-32 后每行实际聚合强度
        topk_vals, _ = g.topk(32, dim=-1)

        stats.append({
            'layer': i,
            'e_abs_mean': e.abs().mean().item(),
            'e_abs_max': e.abs().max().item(),
            'e_saturated_frac': sat,
            'p_norm_per_tok': (p.norm() / math.sqrt(p.numel())).item() if i > 0 else 0.0,
            'g_mean': g.mean().item(),
            'g_std': g.std().item(),
            'g_gt0.9_frac': (g > 0.9).float().mean().item(),
            'g_lt0.1_frac': (g < 0.1).float().mean().item(),
            'topk32_mean': topk_vals.mean().item(),
        })
    return stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    print(f"Device: {DEV}")
    ckpt_path = Path('results/pcn_v1/best_model.pt')
    model = make_pcn().to(DEV)
    sd = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if missing or unexpected:
        print(f"  ⚠️  {missing[:3]} ... / {unexpected[:3]} ...")
    model.eval()

    ids = random_batch(4, 128, seed=42).to(DEV)

    # ============ 前向：抓每层输入 ============
    layer_inputs = {}

    def make_pre_hook(idx):
        def hook(module, args):
            layer_inputs[idx] = args[0].detach().clone()
        return hook

    handles = [blk.register_forward_pre_hook(make_pre_hook(i))
               for i, blk in enumerate(model.layers)]

    with torch.no_grad():
        logits = model(ids, topk=32, no_feedback=False, no_gating=False)
    for h in handles:
        h.remove()

    loss_ordered = lm_loss(logits, ids).item()

    # ============ S5 上下文敏感性：打乱每行 token 顺序 ============
    perm = torch.argsort(torch.randn(ids.shape, device=DEV), dim=1)
    ids_shuf = torch.gather(ids, 1, perm)
    with torch.no_grad():
        logits_shuf = model(ids_shuf, topk=32)
    loss_shuffled = lm_loss(logits_shuf, ids_shuf).item()
    ctx_sensitivity = loss_shuffled - loss_ordered  # >0 才正常（用了上下文）

    # ============ 前向统计：每层输入范数 + 误差/门控 ============
    fwd = []
    for i, x in layer_inputs.items():
        fwd.append({
            'layer': i,
            'x_norm_per_tok': (x.norm() / math.sqrt(x.numel())).item(),
        })
    err_stats = recompute_errors(model, [layer_inputs[i] for i in range(len(layer_inputs))])

    # ============ 反向：梯度流 ============
    for p in model.parameters():
        p.grad = None
    layer_inputs.clear()
    grad_of_input = {}
    layer_inputs_list = []

    def make_hook(idx):
        def hook(module, args):
            t = args[0]
            t.retain_grad()
            layer_inputs_list.append((idx, t))
        return hook

    handles = [blk.register_forward_pre_hook(make_hook(i))
               for i, blk in enumerate(model.layers)]
    logits = model(ids, topk=32, no_feedback=False, no_gating=False)
    loss = lm_loss(logits, ids)
    loss.backward()
    for h in handles:
        h.remove()

    grad_flow = []
    for idx, t in layer_inputs_list:
        gn = t.grad.norm().item() if t.grad is not None else 0.0
        grad_flow.append({
            'layer': idx,
            'input_grad_norm': gn,
            'input_grad_per_tok': gn / math.sqrt(t.numel()),
        })

    # 参数梯度：按层 × 模块
    param_grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split('.')
        if len(parts) >= 4 and parts[0] == 'layers':
            layer_idx, mod = parts[1], parts[3]
        else:
            layer_idx, mod = '-', name
        param_grads.append({
            'layer': layer_idx, 'module': mod,
            'grad_norm': p.grad.norm().item(),
            'weight_norm': p.norm().item(),
        })

    # ============ 汇总报告 ============
    report['device'] = str(DEV)
    report['loss_random_tokens_ordered'] = loss_ordered
    report['loss_random_tokens_shuffled'] = loss_shuffled
    report['context_sensitivity'] = ctx_sensitivity
    report['forward_input_norms'] = fwd
    report['error_and_gate_stats'] = err_stats
    report['input_grad_flow'] = grad_flow
    report['param_grads'] = param_grads

    # ============ 控制台摘要 ============
    print("\n" + "=" * 72)
    print("【S5】上下文敏感性（打乱后 loss 应显著变差）")
    print(f"  loss(原序)={loss_ordered:.4f}  loss(打乱)={loss_shuffled:.4f}  Δ={ctx_sensitivity:.4f}")
    print("  " + ("⚠️ 实锤：模型不使用上下文（纯 unigram）" if abs(ctx_sensitivity) < 0.05
                  else "✅ 模型对上下文有一定敏感性"))

    print("\n【S1/S4】每层输入范数（前向信号流）与输入梯度范数（反向梯度流）")
    print(f"  {'层':>4} {'‖x_l‖/√n':>10} {'‖∂L/∂x_l‖':>12}")
    for f_, g_ in zip(fwd, grad_flow):
        print(f"  {f_['layer']:>4} {f_['x_norm_per_tok']:>10.4f} {g_['input_grad_norm']:>12.4e}")
    g0 = grad_flow[0]['input_grad_norm']
    g11 = grad_flow[-1]['input_grad_norm']
    print(f"  梯度比 (层0/层11) = {g0 / max(g11, 1e-12):.3e}"
          + ("  ⚠️ 严重梯度衰减" if g0 / max(g11, 1e-12) < 1e-3 else ""))

    print("\n【误差/门控健康度】")
    print(f"  {'层':>4} {'|e|均值':>8} {'饱和率':>7} {'g均值':>7} {'g_std':>7} {'top32均值':>9}")
    for s in err_stats:
        print(f"  {s['layer']:>4} {s['e_abs_mean']:>8.4f} {s['e_saturated_frac']:>7.4f} "
              f"{s['g_mean']:>7.4f} {s['g_std']:>7.4f} {s['topk32_mean']:>9.4f}")

    print("\n【关键参数梯度】(层: 模块 grad_norm / weight_norm)")
    keys = ('W_res', 'W_td', 'w_g', 'W_lat', 'W_compress')
    for s in param_grads:
        if s['module'] in keys and s['layer'] in ('0', '5', '11'):
            print(f"  L{s['layer']:>2} {s['module']:<12} {s['grad_norm']:>10.4e} / {s['weight_norm']:.4f}")

    with open(OUT_DIR / 'step1_diagnosis.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n完整报告: {OUT_DIR / 'step1_diagnosis.json'}")


if __name__ == '__main__':
    main()
