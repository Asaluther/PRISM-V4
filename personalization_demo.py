#!/usr/bin/env python3
"""V6 端侧 Demo — 个人风格助手：少量数据快速个性化微调

场景：用户给几段话（500-2000 token），模型快速适应其风格。
对照：PCN（误差流，no_gating）vs Transformer，同预算冻结主干微调。
预期：PCN 在少数据适应上有显著优势（二维规律 +84% 象限）。

用法：python personalization_demo.py
"""
import sys, json, time, math, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/demo')


def make_user_data(num_users=5, tokens_per_user=1500):
    """从 TinyStories 不同子集模拟不同'用户'的风格文本"""
    train_loader, _, tok = get_dataloaders(seq_len=256, batch_size=32,
                                            num_workers=0, max_train=50000, max_val=100)
    # 取前 num_users*2 个 batch 作为不同用户的风格样本
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= num_users * 2:
            break
        batches.append((x, y))

    users = []
    for u in range(num_users):
        # 每个"用户"取一个 batch 的一部分，切成 256 长度的块
        x, y = batches[u * 2]
        xs, ys = [], []
        for i in range(x.size(0)):
            if sum(t.numel() for t in xs) >= tokens_per_user:
                break
            xs.append(x[i:i+1])  # [1, 256]
            ys.append(y[i:i+1])
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys),
                      'n_tokens': sum(t.numel() for t in xs)})

    # 测试集：每个用户的"held-out"文本（下一个 batch）
    tests = []
    for u in range(num_users):
        x, y = batches[u * 2 + 1]
        tests.append({'x': x[:4], 'y': y[:4]})  # 4 条 held-out

    return users, tests, tok


@torch.no_grad()
def eval_ppl(model, data, kind):
    model.eval()
    tot, n = 0.0, 0
    for x, y in data:
        x, y = x.to(DEV), y.to(DEV)
        if kind == 'pcn':
            logits = model(x, topk=64, no_gating=True)
        else:
            logits = model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def finetune(model, kind, user_data, steps=300, lr=5e-4, freeze_backbone=True):
    """冻结主干（embedding+前半层），只微调后半层+LM head"""
    if freeze_backbone:
        # 冻结 embedding + 前 6 层
        model.token_emb.weight.requires_grad = False
        model.pos_emb.weight.requires_grad = False
        for i, blk in enumerate(model.layers):
            if i < 6:
                for p in blk.parameters():
                    p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    xs = user_data['x'].to(DEV)
    ys = user_data['y'].to(DEV)
    n_chunks = xs.size(0)
    model.train()
    t0 = time.time()
    for step in range(steps):
        ci = step % n_chunks
        x, y = xs[ci:ci+1], ys[ci:ci+1]  # [1, 256]
        if kind == 'pcn':
            logits = model(x, topk=64, no_gating=True)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 100 == 0:
            print(f'    step {step}: loss={loss.item():.4f}')

    elapsed = time.time() - t0
    return model, elapsed


def run_user_experiment(tag, build_model, kind, ckpt, user, test, steps=300):
    """单用户：加载 checkpoint → 评估初始 → 微调 → 评估改善"""
    print(f'  [{tag}] 用户数据: {user["n_tokens"]} tokens')

    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model = model.to(DEV)

    ppl_before = eval_ppl(model, [(test['x'], test['y'])], kind)
    print(f'    微调前 PPL: {ppl_before:.2f}')

    model_ft, elapsed = finetune(model, kind, user, steps=steps)
    ppl_after = eval_ppl(model_ft, [(test['x'], test['y'])], kind)
    improvement = (1 - ppl_after / ppl_before) * 100
    print(f'    微调后 PPL: {ppl_after:.2f} (改善 {improvement:+.1f}%, 耗时 {elapsed:.1f}s)')

    return {'before': round(ppl_before, 2), 'after': round(ppl_after, 2),
            'improvement_pct': round(improvement, 1), 'time_s': round(elapsed, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== V6 端侧 Demo: 个人风格助手 =====')
    print('场景：用户给几段话（~1500 token），模型快速适应\n')

    # 1. 构建用户数据
    print('[1] 构建合成用户数据...')
    users, tests, tok = make_user_data()
    print(f'  {len(users)} 个用户，每人 ~{users[0]["n_tokens"]} tokens\n')

    # 2. 对照实验
    print('[2] 对照实验：PCN vs Transformer（300 步微调，冻结前半层）\n')

    results = {'pcn': [], 'transformer': []}
    for u in range(len(users)):
        print(f'--- 用户 {u+1} ---')

        r_pcn = run_user_experiment(
            'PCN', lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                     n_heads=4, d_gate=64, dropout=0.0,
                                     max_seq_len=256, init_mode='fixed'),
            'pcn', 'results/wt_causal_no_gating_s0/best_model.pt',
            users[u], tests[u])
        results['pcn'].append(r_pcn)

        r_tf = run_user_experiment(
            'TF', lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                            n_heads=4, ffn_dim=1024, dropout=0.0,
                                            max_seq_len=256, attn_impl='mha'),
            'tf', 'results/wt_causal_transformer_s0/best_model.pt',
            users[u], tests[u])
        results['transformer'].append(r_tf)
        print()

    # 3. 汇总
    print('===== 汇总 =====')
    for kind in ('pcn', 'transformer'):
        imps = [r['improvement_pct'] for r in results[kind]]
        times = [r['time_s'] for r in results[kind]]
        avg_imp = sum(imps) / len(imps)
        avg_time = sum(times) / len(times)
        print(f'  {kind:>12}: 平均 PPL 改善 {avg_imp:+.1f}%  平均耗时 {avg_time:.1f}s')

    # PCN 优势
    pcn_imps = [r['improvement_pct'] for r in results['pcn']]
    tf_imps = [r['improvement_pct'] for r in results['transformer']]
    wins = sum(1 for p, t in zip(pcn_imps, tf_imps) if p > t)
    print(f'\n  PCN 胜出: {wins}/{len(users)} 用户')

    json.dump(results, open(OUT / 'personalization_demo.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/personalization_demo.json')


if __name__ == '__main__':
    main()
