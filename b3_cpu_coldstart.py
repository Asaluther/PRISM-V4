#!/usr/bin/env python3
"""B3 — CPU 冷启动适应：无 GPU 环境下的少量数据微调

问题：GPU 上冷启动需 4.6 秒。CPU 上能否在 30 秒内完成？（终端判据）
"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.data.tinystories import get_dataloaders

VOCAB = 50257
CKPT = 'results/wt_causal_no_gating_s0/best_model.pt'
OUT = Path('results/v7')
STEPS = 300


def make_data():
    train_loader, _, _ = get_dataloaders(seq_len=256, batch_size=32,
                                          num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        batches.append((x, y))
    users = []
    for u in range(3):  # 3 用户即可验证
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
    tests = []
    for u in range(3):
        tests.append({'x': batches[u*2+1][0][:2], 'y': batches[u*2+1][1][:2]})
    return users, tests


@torch.no_grad()
def eval_ppl(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1], data['y'][i:i+1]
        logits = model(x, topk=64, no_gating=True)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def finetune_cpu(model, train_data, steps=STEPS):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = train_data['x'], train_data['y']
    model.train()
    t0 = time.time()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, topk=64, no_gating=True)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    elapsed = time.time() - t0
    model.eval()
    return model, elapsed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== B3: CPU 冷启动适应 =====\n')
    users, tests = make_data()

    results = []
    for u in range(len(users)):
        m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                     d_gate=64, dropout=0.0, max_seq_len=256,
                     init_mode='fixed', act_sparse=0.0)
        m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
        m = m.eval()

        pb = eval_ppl(m, tests[u])
        m_ft, elapsed = finetune_cpu(m, users[u])
        pa = eval_ppl(m_ft, tests[u])
        imp = (1 - pa / pb) * 100
        print(f'  用户{u+1}: PPL {pb:.0f}→{pa:.0f}  改善 {imp:+.1f}%  耗时 {elapsed:.1f}s')
        results.append({'user': u+1, 'before': round(pb,1), 'after': round(pa,1),
                        'improvement': round(imp,1), 'time_s': round(elapsed,1)})

    avg_imp = sum(r['improvement'] for r in results) / len(results)
    avg_time = sum(r['time_s'] for r in results) / len(results)
    print(f'\n平均: 改善 {avg_imp:+.1f}%  耗时 {avg_time:.1f}s')
    print(f'判定: {"PASS ✅" if avg_time <= 30 and avg_imp > 0 else "FAIL ❌"} '
          f'(目标 ≤30s 且改善>0)')

    json.dump(results, open(OUT / 'b3_cpu_coldstart.json', 'w'), indent=2)
    print(f'saved -> {OUT}/b3_cpu_coldstart.json')


if __name__ == '__main__':
    main()
