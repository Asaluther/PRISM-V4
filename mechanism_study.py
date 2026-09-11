#!/usr/bin/env python3
"""机理理论化 E1-E3：为什么误差流泛化而 TF 过拟合？

E1: 权重变化分析（各层 ΔW 范数分布）
E2: 表示距离分析（训练块 vs held-out 的余弦距离）
E3: 误差减法因果实验（e=u vs e=u−p 的泛化对比）
"""
import sys, json, math, copy, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/mechanism')
STEPS = 300


def make_data():
    train_loader, _, tok = get_dataloaders(seq_len=256, batch_size=32,
                                            num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 2:
            break
        batches.append((x, y))
    train = {'x': batches[0][0][:6], 'y': batches[0][1][:6]}
    heldout = {'x': batches[1][0][:4], 'y': batches[1][1][:4]}
    return train, heldout, tok


def freeze_and_finetune(model, kind, train_data, steps=STEPS, lr=5e-4):
    """冻结前 6 层 + embedding，微调后半，返回微调前后的权重快照"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # 快照微调前权重
    pre_weights = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            pre_weights[name] = p.detach().clone()

    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        if kind == 'pcn':
            logits = model(x, topk=64, no_gating=True)
        elif kind == 'pcn_nf':
            logits = model(x, topk=64, no_gating=True, no_feedback=True)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()

    # 计算各层 ΔW 范数
    deltas = {}
    for name, p in model.named_parameters():
        if name in pre_weights:
            deltas[name] = (p.detach() - pre_weights[name]).norm().item()
    return model, deltas


@torch.no_grad()
def eval_ppl(model, data, kind):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        if kind == 'pcn':
            logits = model(x, topk=64, no_gating=True)
        elif kind == 'pcn_nf':
            logits = model(x, topk=64, no_gating=True, no_feedback=True)
        else:
            logits = model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


@torch.no_grad()
def extract_layer_reps(model, x, kind):
    """提取各层输出表示"""
    reps = []
    hooks = []
    def hook_fn(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        reps.append(t.detach().cpu())
    for blk in model.layers:
        hooks.append(blk.register_forward_hook(hook_fn))
    with torch.no_grad():
        if kind in ('pcn', 'pcn_nf'):
            kwargs = {'topk': 64, 'no_gating': True}
            if kind == 'pcn_nf':
                kwargs['no_feedback'] = True
            model(x.to(DEV), **kwargs)
        else:
            model(x.to(DEV))
    for h in hooks:
        h.remove()
    return reps  # list of [B, N, D]


def rep_distance(reps_train, reps_held):
    """训练块与 held-out 的各层余弦距离（越大=分化越严重=过拟合信号）"""
    dists = []
    for rt, rh in zip(reps_train, reps_held):
        # 平均 batch 维度 → [N, D]，然后展平
        ft = rt.mean(0).flatten()
        fh = rh.mean(0).flatten()
        cos = F.cosine_similarity(ft, fh, dim=0).item()
        dists.append(1.0 - cos)
    return dists


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== 机理理论化：误差流泛化 vs TF 过拟合 =====\n')
    train, heldout, tok = make_data()

    # ===== E3（先跑：因果实验最重要）=====
    print('===== E3: 误差减法因果实验 =====')
    print('  PCN (e=u−p, 有减法) vs PCN-nf (e=u, 无减法) vs TF')
    print('  同一用户数据微调 300 步，比较 held-out PPL 变化\n')

    e3 = {}
    for tag, kind, build, ckpt in (
        ('PCN (e=u-p)', 'pcn',
         lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                          n_heads=4, d_gate=64, dropout=0.0, max_seq_len=256,
                          init_mode='fixed'),
         'results/wt_causal_no_gating_s0/best_model.pt'),
        ('PCN-nf (e=u)', 'pcn_nf',
         lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                          n_heads=4, d_gate=64, dropout=0.0, max_seq_len=256,
                          init_mode='fixed'),
         'results/wt_causal_no_gating_s0/best_model.pt'),
        ('TF', 'tf',
         lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                   n_heads=4, ffn_dim=1024, dropout=0.0,
                                   max_seq_len=256, attn_impl='mha'),
         'results/wt_causal_transformer_s0/best_model.pt'),
    ):
        m = build()
        m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
        m = m.to(DEV).eval()

        ppl_before = eval_ppl(m, heldout, kind)
        m_ft, deltas = freeze_and_finetune(m, kind, train)
        ppl_after = eval_ppl(m_ft, heldout, kind)
        change = (1 - ppl_after / ppl_before) * 100

        print(f'  {tag:<16} before={ppl_before:.1f}  after={ppl_after:.1f}  '
              f'{"✅ 改善" if change > 0 else "❌ 恶化"} {change:+.1f}%')

        # E1: 权重变化按层汇总
        layer_deltas = {}
        for name, d in deltas.items():
            parts = name.split('.')
            if parts[0] == 'layers':
                li = int(parts[1])
                layer_deltas.setdefault(li, []).append(d)
        e1_summary = {li: round(sum(v) / len(v), 6) for li, v in sorted(layer_deltas.items())}

        # E2: 表示距离
        reps_train_pre = extract_layer_reps(m, train['x'][:2], kind)
        reps_held_pre = extract_layer_reps(m, heldout['x'][:2], kind)
        dist_pre = rep_distance(reps_train_pre, reps_held_pre)
        reps_train_post = extract_layer_reps(m_ft, train['x'][:2], kind)
        reps_held_post = extract_layer_reps(m_ft, heldout['x'][:2], kind)
        dist_post = rep_distance(reps_train_post, reps_held_post)

        e3[tag] = {'before': round(ppl_before, 1), 'after': round(ppl_after, 1),
                   'change_pct': round(change, 1),
                   'e1_layer_deltas': e1_summary,
                   'e2_dist_pre': [round(d, 4) for d in dist_pre],
                   'e2_dist_post': [round(d, 4) for d in dist_post]}

    # ===== 判定 =====
    print('\n===== E1: 各层权重变化（后半层） =====')
    for tag in e3:
        ld = e3[tag]['e1_layer_deltas']
        back_half = [ld.get(i, 0) for i in range(6, 12)]
        vals_str = ', '.join(f'{v:.4f}' for v in back_half)
        print(f'  {tag:<16} 后半层 ΔW: [{vals_str}]')

    print('\n===== E2: 表示距离变化（Δdist = post − pre，正=分化加剧） =====')
    for tag in e3:
        dp = e3[tag]['e2_dist_post']
        dpre = e3[tag]['e2_dist_pre']
        delta_dist = [round(p - r, 4) for p, r in zip(dp, dpre)]
        print(f'  {tag:<16} Δdist: {delta_dist}')

    print('\n===== E3 判定 =====')
    pcn_c = e3['PCN (e=u-p)']['change_pct']
    nf_c = e3['PCN-nf (e=u)']['change_pct']
    tf_c = e3['TF']['change_pct']
    if nf_c < 0 and pcn_c > 0:
        print(f'  ✅ 假说证实：误差减法(e=u−p)是泛化的关键——去掉减法(e=u)后也过拟合')
    elif nf_c > 0 and pcn_c > 0:
        print(f'  ⚠️ 误差减法非必要——无减法版也泛化，假说被否定')
    else:
        print(f'  ❓ 混合结果，需进一步分析')

    json.dump(e3, open(OUT / 'generalization_mechanism.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/generalization_mechanism.json')


if __name__ == '__main__':
    main()
