#!/usr/bin/env python3
"""A4 — 稀疏 × 冷启动交互：稀疏化后 PCN 的冷启动优势是否保持？

问题：A2 发现 25% 稀疏改善 PPL 14%，但终端场景的核心价值是冷启动适应。
     如果稀疏化损害了冷启动能力，则终端部署需要权衡。
协议：复用 personalization_demo 的微调流程，比较三种模型的冷启动表现：
  1. PCN 稠密（act_sparse=0）
  2. PCN 稀疏 25%（act_sparse=0.25）
  3. PCN 稀疏 50%（act_sparse=0.50）
"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.data.tinystories import get_dataloaders

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/v7')
STEPS = 300


def make_data():
    train_loader, _, tok = get_dataloaders(seq_len=256, batch_size=32,
                                            num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        batches.append((x, y))
    # 5 个"用户"
    users = []
    for u in range(5):
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
    tests = []
    for u in range(5):
        tests.append({'x': batches[u*2+1][0][:4], 'y': batches[u*2+1][1][:4]})
    return users, tests


@torch.no_grad()
def eval_ppl(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, topk=64, no_gating=True)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def finetune(model, train_data, steps=STEPS):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, topk=64, no_gating=True)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== A4: 稀疏 × 冷启动交互 =====\n')
    users, tests = make_data()

    # 训练 checkpoint（用 A2 扫描中已有的最好结果）
    # sparse_0.25_s0 和 sparse_0.50_s0（都是 ~14.1 PPL）
    ckpts = {
        'dense (0%)': ('results/wt_causal_no_gating_s0/best_model.pt', 0.0),
        'sparse 25%': ('results/sparse_0.25_s0/best_model.pt', 0.25),
        'sparse 50%': ('results/sparse_0.50_s0/best_model.pt', 0.50),
    }

    all_results = {}
    for tag, (ckpt, sp) in ckpts.items():
        print(f'--- {tag} ---')
        improvements = []
        for u in range(len(users)):
            m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                         d_gate=64, dropout=0.0, max_seq_len=256,
                         init_mode='fixed', act_sparse=sp)
            m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
            m = m.to(DEV)
            pb = eval_ppl(m, tests[u])
            m_ft = finetune(m, users[u])
            pa = eval_ppl(m_ft, tests[u])
            imp = (1 - pa / pb) * 100
            improvements.append(imp)
        avg = sum(improvements) / len(improvements)
        wins = sum(1 for i in improvements if i > 0)
        print(f'  平均改善: {avg:+.1f}%  正改善: {wins}/{len(users)}')
        all_results[tag] = {'improvements': [round(i,1) for i in improvements],
                            'avg': round(avg,1), 'wins': f'{wins}/{len(users)}'}

    # 判定
    print('\n===== 判定 =====')
    dense_avg = all_results['dense (0%)']['avg']
    for tag, r in all_results.items():
        ratio = r['avg'] / dense_avg if dense_avg != 0 else 0
        keep = '✅ 优势保持' if r['avg'] > dense_avg * 0.5 else '⚠️ 优势受损'
        print(f'  {tag:<14} 冷启动 {r["avg"]:+.1f}%  (稠密的 {ratio:.2f}x)  {keep}')

    json.dump(all_results, open(OUT / 'a4_sparse_coldstart.json', 'w'), indent=2)
    print(f'\nsaved -> {OUT}/a4_sparse_coldstart.json')


if __name__ == '__main__':
    main()
