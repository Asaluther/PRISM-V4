#!/usr/bin/env python3
"""PRISM V4 — 证据线 A：行为签名分析

V4_PLAN §2 的判定指标：
  1. 层间 CKA：PCN-fixed vs Transformer 的内部表示结构是否定性不同
  2. 误差空间分布（PCN 专属）：预测误差范数与 token「惊讶度」的相关性
     —— 预测编码假说：误差应集中在 surprising token（低频/句法边界）
  3. 聚合熵对比：PCN 门控分布熵 vs Transformer attention 熵
  4. 训练动态：val PPL 曲线形状对比（从 results.json 读取）

输入：results/pcn_v2_fixed 与 results/transformer_baseline 的 best_model.pt
输出：results/analysis_v2/signatures.json + signatures.md
"""

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders
from src.analysis.cka import cka_matrix

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('results/analysis_v2')
SEQ = 256


def load_pcn(ckpt, d_gate=32):
    m = PCNModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                 d_gate=d_gate, dropout=0.1, max_seq_len=SEQ, init_mode='fixed')
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV).eval()


def load_tf(ckpt):
    m = TransformerModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                         ffn_dim=1024, dropout=0.1, max_seq_len=256, attn_impl='mha')
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV).eval()


@torch.no_grad()
def collect(model, kind, loader, n_batches=4, topk=32, feedback_mode='prev_layer', no_gating=False):
    """返回每层输出 [rows, D]、每位置 loss、PCN 门控熵 / TF attention 熵"""
    feats = {}
    attn_entropies = []

    def feat_hook(idx):
        def fn(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            feats.setdefault(idx, []).append(t.detach().float().cpu())
        return fn

    def attn_hook(idx):
        def fn(module, args, output):
            w = output[1]  # MultiheadAttention(with need_weights=True): [B, N, N]
            if w is not None:
                p = w.detach().float()
                ent = -(p * (p + 1e-10).log()).sum(-1)  # [B, N]
                attn_entropies.append(ent.mean().item())
        return fn

    hs = [blk.register_forward_hook(feat_hook(i)) for i, blk in enumerate(model.layers)]
    has = []
    if kind == 'transformer' or (kind == 'pcn' and no_gating):
        has = [blk.attn.register_forward_hook(attn_hook(i))
               for i, blk in enumerate(model.layers)]

    losses = []
    inputs = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x, y = batch[0].to(DEV), batch[1].to(DEV)
            inputs.append((x, y))
            if kind == 'pcn':
                kwargs = dict(topk=topk, feedback_mode=feedback_mode)
                if no_gating:
                    kwargs['no_gating'] = True
                logits = model(x, **kwargs)
            else:
                logits = model(x)
            l = F.cross_entropy(logits[:, :-1].reshape(-1, 50257),
                                y[:, 1:].reshape(-1), reduction='none')
            losses.append(l.reshape(x.size(0), -1).cpu())  # [B, N-1]
    for h in hs + has:
        h.remove()

    layers = [torch.cat(v, dim=0).reshape(-1, v[0].shape[-1]) for v in
              (feats[i] for i in sorted(feats))]
    return layers, torch.cat(losses), inputs, attn_entropies


@torch.no_grad()
def pcn_gate_entropy_and_errors(model, inputs, topk=32, feedback_mode='prev_layer'):
    """复算每层误差范数（与 token 惊讶度做相关）和门控熵
    two_pass 模式下 pre-hook 会先抓到 pass1 再被 pass2 覆盖——最终是 pass2 输入，正确"""
    x, y = inputs[0]
    captured = {}

    def pre(idx):
        def fn(module, args):
            captured[idx] = args[0].detach()
        return fn

    hs = [b.register_forward_pre_hook(pre(i)) for i, b in enumerate(model.layers)]
    with torch.no_grad():
        model(x, topk=topk, feedback_mode=feedback_mode)
    for h in hs:
        h.remove()

    err_norms, gate_ent = [], []
    for i, blk in enumerate(model.layers):
        pc = blk.pcn_layer
        u = pc.W_bu(pc.norm_bu(captured[i]))
        p = pc.W_td(captured[i - 1]) if i > 0 else torch.zeros_like(u)
        e = (u - p).clamp(-10, 10)
        err_norms.append(e.norm(dim=-1).cpu())            # [B, N]

        e_comp = pc.W_compress(e)
        if pc.gate_ln:
            e_comp = F.layer_norm(e_comp, (e_comp.size(-1),))
        e_i, e_j = e_comp.unsqueeze(2), e_comp.unsqueeze(1)
        inter = torch.cat([e_i * e_j, e_i + e_j], dim=-1)
        g = torch.sigmoid(pc.w_g(inter).squeeze(-1))      # [B, N, N]
        # 与训练一致：因果掩码后再做 Top-K
        causal = torch.tril(torch.ones(g.size(-1), g.size(-1),
                                       device=g.device, dtype=torch.bool))
        g = g.masked_fill(~causal, 0.0)
        tv, ti = g.topk(topk, dim=-1)
        dist = torch.zeros_like(g).scatter_(-1, ti, tv)
        dist = dist / (dist.sum(-1, keepdim=True) + 1e-10)
        gate_ent.append((-(dist * (dist + 1e-10).log()).sum(-1)).mean().item())

    # 误差范数沿层平均，取 [B, N-1] 与逐位置 loss 对齐
    err_avg = torch.stack(err_norms).mean(0)[:, :-1].reshape(-1)
    loss_flat = F.cross_entropy  # noqa
    return err_avg.reshape(-1), gate_ent


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-10)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcn_ckpt', default='results/pcn_v2_fixed/best_model.pt')
    ap.add_argument('--tag', default='fixed')
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--d_gate', type=int, default=32)
    ap.add_argument('--no_gating', action='store_true',
                    help='checkpoint 是 no_gating 消融（attention 聚合）')
    ap.add_argument('--tf_ckpt', default='results/ts_causal_transformer_s0/best_model.pt')
    ap.add_argument('--feedback_mode', default='prev_layer',
                    choices=['prev_layer', 'two_pass'])
    ns = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    _, val_loader, _ = get_dataloaders(seq_len=SEQ, batch_size=4, max_train=50000, max_val=5000)

    print(f"Loading models... (pcn ckpt: {ns.pcn_ckpt})")
    pcn = load_pcn(ns.pcn_ckpt, d_gate=ns.d_gate)
    tf = load_tf(ns.tf_ckpt)

    print("Collecting PCN features...")
    pcn_layers, pcn_loss, pcn_inputs, _ = collect(pcn, 'pcn', val_loader,
                                                  topk=ns.topk,
                                                  feedback_mode=ns.feedback_mode,
                                                  no_gating=ns.no_gating)
    print("Collecting Transformer features...")
    tf_layers, tf_loss, _, tf_attn_ent = collect(tf, 'transformer', val_loader)

    # ---- 1. CKA ----
    print("CKA matrices...")
    sub = 2048
    pcn_f = [l[:sub] for l in pcn_layers]
    tf_f = [l[:sub] for l in tf_layers]
    cka_pcn = cka_matrix([f.cuda() for f in pcn_f], [f.cuda() for f in pcn_f]).cpu()
    cka_tf = cka_matrix([f.cuda() for f in tf_f], [f.cuda() for f in tf_f]).cpu()
    cka_cross = cka_matrix([f.cuda() for f in pcn_f], [f.cuda() for f in tf_f]).cpu()

    # ---- 2. 误差 vs 惊讶度 ----
    print("Error-surprise correlation...")
    if ns.no_gating:
        # no_gating 消融：pcn_layer 的门控权重未参与训练，门控熵无意义，跳过
        x0, _ = pcn_inputs[0]
        captured = {}

        def pre(idx):
            def fn(module, args):
                captured[idx] = args[0].detach()
            return fn

        hs = [b.register_forward_pre_hook(pre(i)) for i, b in enumerate(pcn.layers)]
        with torch.no_grad():
            kwargs = dict(topk=ns.topk, feedback_mode=ns.feedback_mode, no_gating=True)
            pcn(x0, **kwargs)
        for h in hs:
            h.remove()
        errs = []
        for i, blk in enumerate(pcn.layers):
            pc_ = blk.pcn_layer
            u = pc_.W_bu(pc_.norm_bu(captured[i]))
            p = pc_.W_td(captured[i - 1]) if i > 0 else torch.zeros_like(u)
            errs.append((u - p).norm(dim=-1).cpu())
        err_avg = torch.stack(errs).mean(0)[:, :-1].reshape(-1)
        gate_ent = []
    else:
        err_avg, gate_ent = pcn_gate_entropy_and_errors(pcn, pcn_inputs,
                                                        topk=ns.topk,
                                                        feedback_mode=ns.feedback_mode)
    pcn_pos_loss = pcn_loss.reshape(-1)[:err_avg.numel()]
    err_surprise_corr = pearson(err_avg, pcn_pos_loss)

    # 对照：输入 token 的 unigram 惊讶度（频次代理）与 TF 表示的相关可后续扩展
    result = {
        'cka_pcn_internal_diag_mean': cka_pcn.diag().mean().item(),
        'cka_tf_internal_diag_mean': cka_tf.diag().mean().item(),
        'cka_cross_max_per_pcn_layer': cka_cross.max(dim=1).values.tolist(),
        'cka_cross_mean': cka_cross.mean().item(),
        'cka_pcn_internal': cka_pcn.tolist(),
        'cka_tf_internal': cka_tf.tolist(),
        'cka_cross': cka_cross.tolist(),
        'error_surprise_pearson': err_surprise_corr,
        'pcn_gate_entropy_per_layer': gate_ent,
        'pcn_gate_entropy_mean': (sum(gate_ent) / len(gate_ent)) if gate_ent else None,
        'tf_attention_entropy_mean_per_layer': tf_attn_ent,
        'tf_attention_entropy_mean': sum(tf_attn_ent) / len(tf_attn_ent) if tf_attn_ent else None,
        'uniform_entropy_lnN': math.log(256),
    }

    # ---- 3. 训练动态对比 ----
    curves = {}
    for name in ('pcn_v2_fixed', 'transformer_baseline', 'pcn_v2_no_feedback', 'pcn_v2_no_gating'):
        p = Path(f'results/{name}/results.json')
        if p.exists():
            r = json.loads(p.read_text())
            curves[name] = {
                'best_ppl': r.get('best_ppl'),
                'val_ppl': [(v['step'], round(v['ppl'], 2)) for v in r.get('val_ppl', [])],
                'budget': r.get('budget'),
            }
    result['training_curves'] = curves

    with open(OUT / f'signatures_{ns.tag}.json', 'w') as f:
        json.dump(result, f, indent=2)

    # ---- 摘要 ----
    ge = result['pcn_gate_entropy_mean']
    lines = [
        f"# PRISM V4 — 行为签名分析（证据线 A | ckpt={ns.tag}）\n",
        f"- 误差-惊讶度 Pearson 相关（PCN, 12 层平均）: **{err_surprise_corr:.3f}**",
        f"  （预测编码假说预测正相关；0 附近=误差与惊讶度无关）",
        f"- PCN 门控熵（归一化分布, nats）: {f'{ge:.3f}' if ge is not None else 'N/A（no_gating 消融无门控）'}"
        f" / 均匀上限 ln256={math.log(256):.3f}",
        f"- Transformer attention 熵: {result['tf_attention_entropy_mean']:.3f}"
        if tf_attn_ent else "- Transformer attention 熵: N/A",
        f"- 跨模型 CKA 均值: {result['cka_cross_mean']:.3f}"
        f"（低=两模型表示结构不同）",
        f"- PCN 内部 CKA 对角均值: {result['cka_pcn_internal_diag_mean']:.3f}",
        f"- TF 内部 CKA 对角均值: {result['cka_tf_internal_diag_mean']:.3f}\n",
        "## 训练曲线\n",
    ]
    for name, c in curves.items():
        lines.append(f"- **{name}**: best PPL={c['best_ppl']:.2f} | 曲线={c['val_ppl']}")
    (OUT / f'signatures_{ns.tag}.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(f"\n输出: {OUT}/signatures_{ns.tag}.json + signatures_{ns.tag}.md")


if __name__ == '__main__':
    main()
