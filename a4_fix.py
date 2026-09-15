#!/usr/bin/env python3
"""A4-fix — 修正版：拆解稀疏效应 vs 域效应

原 A4 的混杂：
  稠密 PCN = WikiText 预训练 → TinyStories 微调 = 跨域（PCN 强项）
  稀疏 PCN = TinyStories 预训练 → TinyStories 微调 = 域内（已知无优势）

修正设计（四臂对照）：
  ① 稠密 × WikiText → TS 微调 = 跨域冷启动（PCN 强项，已有基线）
  ② 稀疏 × WikiText → TS 微调 = 跨域冷启动 + 稀疏（需新训练 WikiText 稀疏模型）
  ③ 稠密 × TS → TS 微调 = 域内微调（已知弱，用 ts_causal_no_gating_s0）
  ④ 稀疏 × TS → TS 微调 = 域内微调 + 稀疏（原 A4 数据，可复用）

判定：
  若 ① >> ② → 稀疏确实摧毁跨域冷启动（原结论成立）
  若 ① ≈ ② → 稀疏不损害冷启动（原结论是域效应伪影）
  ③ vs ④ → 域内场景下稀疏的额外影响
"""
import subprocess, sys, os
from pathlib import Path

W4_ROOT = Path(__file__).parent

# 步骤 1: 检查是否有 WikiText 稀疏模型
def check_ckpt(path):
    return (W4_ROOT / path).exists()

def main():
    print('===== A4-fix: 四臂对照实验 =====\n')

    ckpts = {
        '① dense × WT（跨域）': 'results/wt_causal_no_gating_s0/best_model.pt',
        '② sparse25 × WT（跨域+稀疏）': 'results/wt_sparse_0.25_s0/best_model.pt',
        '③ dense × TS（域内）': 'results/ts_causal_no_gating_s0/best_model.pt',
        '④ sparse25 × TS（域内+稀疏）': 'results/sparse_0.25_s0/best_model.pt',
    }

    need_train = []
    for tag, path in ckpts.items():
        exists = check_ckpt(path)
        print(f'  {tag}: {"✅ 已有" if exists else "❌ 需训练"}')
        if not exists:
            need_train.append((tag, path))

    if need_train:
        print(f'\n需训练 {len(need_train)} 个模型。')
        print('② WikiText 稀疏 25%（关键——拆解混杂的必需品）')
        run = input('\n启动训练？(y/n): ').strip().lower()
        if run == 'y':
            train_wt_sparse()
        else:
            print('跳过训练。请先运行:')
            print('  HF_DATASETS_OFFLINE=1 ... python train.py --model pcn --no_gating '
                  '--act_sparse 0.25 --dataset wikitext ... --exp_name wt_sparse_0.25_s0')
            return

    # 运行四臂冷启动测试
    run_four_arm(ckpts)


def train_wt_sparse():
    """训练 WikiText 稀疏 25% 模型"""
    env = dict(os.environ)
    env.update({
        'HF_DATASETS_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
    })
    cmd = [
        str(W4_ROOT / '.venv/Scripts/python.exe'),
        str(W4_ROOT / 'train.py'),
        '--model', 'pcn', '--no_gating',
        '--act_sparse', '0.25',
        '--dataset', 'wikitext',
        '--d_gate', '64', '--topk', '64',
        '--lr', '5e-4', '--grad_clip', '0.5',
        '--batch_size', '32', '--seq_len', '256', '--max_seq_len', '256',
        '--max_steps', '5000', '--warmup_steps', '500',
        '--weight_decay', '0.1',
        '--eval_interval', '1000', '--log_interval', '1000',
        '--seed', '0',
        '--exp_name', 'wt_sparse_0.25_s0',
    ]
    print(f'训练 WikiText 稀疏模型 (~10 min)...')
    subprocess.run(cmd, cwd=W4_ROOT, env=env)
    print('训练完成')


def run_four_arm(ckpts):
    """四臂冷启动对照"""
    import torch
    import torch.nn.functional as F
    import math
    from src.models.pcn import PCNModel
    from src.data.tinystories import get_dataloaders

    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VOCAB = 50257
    STEPS = 300

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

    def finetune(model, train_data):
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
        for step in range(STEPS):
            ci = step % xs.size(0)
            x, y = xs[ci:ci+1], ys[ci:ci+1]
            logits = model(x, topk=64, no_gating=True)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        model.eval()
        return model

    print('\n===== 四臂冷启动对照 =====\n')

    arm_configs = [
        ('① dense × WT', 'results/wt_causal_no_gating_s0/best_model.pt', 0.0),
        ('② sparse25 × WT', 'results/wt_sparse_0.25_s0/best_model.pt', 0.25),
        ('③ dense × TS', 'results/ts_causal_no_gating_s0/best_model.pt', 0.0),
        ('④ sparse25 × TS', 'results/sparse_0.25_s0/best_model.pt', 0.25),
    ]

    results = {}
    for tag, ckpt, sp in arm_configs:
        if not Path(ckpt).exists():
            print(f'  {tag}: CHECKPOINT MISSING, SKIP')
            continue
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
            improvements.append((1 - pa / pb) * 100)
        avg = sum(improvements) / len(improvements)
        results[tag] = avg
        print(f'  {tag:<22} 平均改善: {avg:+.1f}%  '
              f'(range: {min(improvements):+.1f}% ~ {max(improvements):+.1f}%)')

    # 判定
    print('\n===== 判定 =====')
    if '① dense × WT' in results and '② sparse25 × WT' in results:
        d_wt = results['① dense × WT']
        s_wt = results['② sparse25 × WT']
        gap = d_wt - s_wt
        if gap > 30:
            print(f'  跨域: 稠密 {d_wt:+.1f}% vs 稀疏 {s_wt:+.1f}% (差距 {gap:.1f}pp)')
            print(f'  → 稀疏确实大幅损害跨域冷启动（原结论成立）')
        elif gap < 10:
            print(f'  跨域: 稠密 {d_wt:+.1f}% vs 稀疏 {s_wt:+.1f}% (差距 {gap:.1f}pp)')
            print(f'  → 稀疏不损害冷启动（原 A4 结论为域效应伪影，需修正）')
        else:
            print(f'  跨域: 稠密 {d_wt:+.1f}% vs 稀疏 {s_wt:+.1f}% (差距 {gap:.1f}pp)')
            print(f'  → 稀疏部分损害冷启动（程度中等）')

    if '③ dense × TS' in results and '④ sparse25 × TS' in results:
        d_ts = results['③ dense × TS']
        s_ts = results['④ sparse25 × TS']
        print(f'\n  域内: 稠密 {d_ts:+.1f}% vs 稀疏 {s_ts:+.1f}%')
        print(f'  → 域内场景{"稀疏额外恶化" if s_ts < d_ts - 10 else "两者相当"}')

    import json
    json.dump({k: round(v,1) for k,v in results.items()},
              open('results/v7/a4_fixed.json','w'), indent=2)
    print('\nsaved -> results/v7/a4_fixed.json')


if __name__ == '__main__':
    main()
