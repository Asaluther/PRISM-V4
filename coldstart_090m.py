#!/usr/bin/env python3
"""Coldstart-090M — 90M 规模跨域冷启动判决实验

问题：小规模（12L）验证的误差流冷启动优势（+58.5% vs TF +5.4%）在
     24L/90-101M 上是否仍成立？（终端愿景核心机制的规模外推测试）

设计（沿用 a4_fix 协议，双臂对照）：
  A臂 PCN 90.5M  (WT 预训练, lr 1e-4 最优) → TS 少量数据微调
  B臂 TF 101.5M  (WT 预训练, lr 5e-4 最优) → TS 少量数据微调
  双方对称：冻结 embedding（含绑定输出头）+ 底部 12/24 层（比例同小规模 6/12）；
  AdamW lr 5e-4, wd 0.01, 300 步, clip 1.0；全程 fp32（无 autocast）
  指标：3 个"用户"各自测试集的 PPL 改善百分比

判定：
  PCN 领先 >20pp → 冷启动优势在 90M 保持，混合路径（TF 骨干 + 误差流适应）成立
  两者接近（±20pp）→ 优势被规模抹平，愿景核心机制需重估
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel
from src.data.tinystories import get_dataloaders

VOCAB = 50257
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STEPS = 300
FREEZE_LAYERS = 12   # 底部一半（同小规模协议 6/12 的比例；hybrid 的 layers=TF 骨干，
                    # 恰好全部冻结——适应只发生在 PCN 头，混合架构的设计意图）
OUT = Path('results/v7')

CKPTS = {
    'PCN 90.5M (lr1e-4)': 'results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt',
    'TF 101.5M (lr5e-4)': 'results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
    'HYB 96.0M (12TF+12PCN)': 'results/wt_094m_hyb_s0/best_model.pt',
}

# 与 train.py 训练时的 model_kwargs 完全一致
PCN_KW = dict(topk=128, no_feedback=False, no_gating=True,
              feedback_mode='prev_layer', n_pass=1)


def make_data():
    train_loader, _, _ = get_dataloaders(seq_len=256, batch_size=32,
                                         num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        batches.append((x, y))
    users = []
    for u in range(3):
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
    tests = []
    for u in range(3):
        tests.append({'x': batches[u*2+1][0][:2], 'y': batches[u*2+1][1][:2]})
    return users, tests


@torch.no_grad()
def eval_ppl(model, data, is_pcn):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, **PCN_KW) if is_pcn else model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def finetune(model, train_data, is_pcn):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < FREEZE_LAYERS:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(STEPS):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, **PCN_KW) if is_pcn else model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def build_model(kind):
    """kind: 'pcn' / 'tf' / 'hyb'（兼容旧布尔调用：True→pcn, False→tf）"""
    if kind is True:
        kind = 'pcn'
    elif kind is False:
        kind = 'tf'
    if kind == 'pcn':
        return PCNModel(vocab_size=VOCAB, d_model=512, n_layers=24, n_heads=4,
                        d_gate=128, dropout=0.0, max_seq_len=256,
                        init_mode='fixed', use_bmm_gate=True)
    if kind == 'hyb':
        return HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                           n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                           d_gate=128, dropout=0.0, max_seq_len=256)
    return TransformerModel(vocab_size=VOCAB, d_model=512, n_layers=24, n_heads=4,
                            ffn_dim=2048, dropout=0.0, max_seq_len=256,
                            attn_impl='sdpa')


def main():
    print(f'Device: {DEV} | 协议: 冻结 emb+底部{FREEZE_LAYERS}/24层 | '
          f'{STEPS} 步 @ lr 5e-4 | fp32')
    users, tests = make_data()
    results, raw = {}, {}
    for tag, ckpt in CKPTS.items():
        if not Path(ckpt).exists():
            print(f'  {tag}: CHECKPOINT MISSING ({ckpt}), SKIP')
            continue
        kind = 'pcn' if tag.startswith('PCN') else ('hyb' if tag.startswith('HYB') else 'tf')
        use_kw = kind != 'tf'   # PCN / HYB 前向需要 PCN kwargs
        improvements, before_ppls, after_ppls = [], [], []
        for u in range(len(users)):
            m = build_model(kind)
            m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
            m = m.to(DEV)
            pb = eval_ppl(m, tests[u], use_kw)
            m_ft = finetune(m, users[u], use_kw)
            pa = eval_ppl(m_ft, tests[u], use_kw)
            improvements.append((1 - pa / pb) * 100)
            before_ppls.append(pb); after_ppls.append(pa)
            print(f'  {tag} user{u}: PPL {pb:.1f} -> {pa:.1f}  ({improvements[-1]:+.1f}%)')
        avg = sum(improvements) / len(improvements)
        results[tag] = avg
        raw[tag] = {'before': before_ppls, 'after': after_ppls,
                    'improvements': improvements, 'avg': avg}
        print(f'  {tag:<22} 平均改善: {avg:+.1f}%  '
              f'(range: {min(improvements):+.1f}% ~ {max(improvements):+.1f}%)\n')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'coldstart_090m.json').write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f'输出: {OUT / "coldstart_090m.json"}')

    print('\n===== 判定 =====')
    if len(results) >= 2:
        pcn_avg = next((v for k, v in results.items() if k.startswith('PCN')), None)
        tf_avg = next((v for k, v in results.items() if k.startswith('TF')), None)
        hyb_avg = next((v for k, v in results.items() if k.startswith('HYB')), None)
        if pcn_avg is not None and tf_avg is not None:
            print(f'  PCN {pcn_avg:+.1f}% vs TF {tf_avg:+.1f}%  (差 {pcn_avg - tf_avg:+.1f}pp)')
        if hyb_avg is not None:
            # 预注册判据（HYBRID_MVP 计划）：单发 ≥+30% → PCN 头适应能力大体保留
            ok = hyb_avg >= 30
            print(f'  HYB {hyb_avg:+.1f}%  预注册判据 ≥+30%: {"✅ 命中" if ok else "❌ 未达"}')


if __name__ == '__main__':
    main()
