#!/usr/bin/env python3
"""step50_mechanism — few-shot 适应 step50 达峰/回落机制分解（纯 CPU, fp32）

阶段③发现：3 用户×6序列、AdamW 5e-4、冻结全部 TF 骨干，改善 step50 达峰(+85%)
后单调回落至 step300(+60%)。报告中的「过拟合 6 条训练序列」解释未经证实——
本脚本把评估点加密到每 10 步，同步追踪四条曲线以区分三个假设：

  H1 过拟合/记忆：train PPL 持续降，test 回升（train-test 缺口在峰值后张开）
  H2 动力学不稳定：train 与 test 同步回升（误差正反馈自毁区间）
  H3 通用遗忘：general PPL（WikiText 探针）显著恶化

附加 lr 扫描 {5e-4, 2.5e-4, 1e-4}：若峰值随 lr 降低而右移/消失，指向漂移速率
驱动（与 H1/H2 相容与否由 train-test 缺口判定）。

协议对齐 phase3_edge（同数据构造、同冻结语义、同优化器/步幅/梯度裁剪）。
用户抽样新固定种子（torch.manual_seed(0)）——与 phase3 的用户不同但同分布，
曲线形状结论不依赖具体用户。

用法：.venv/Scripts/python.exe step50_mechanism.py
输出：results/v7/step50_mechanism.json（每完成一个 (lr,user) 增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.hybrid import HybridModel
from src.data.tinystories import get_dataloaders
from src.data.wikitext import WikiTextDataset

VOCAB = 50257
CKPT = 'results/wt_094m_hyb_s0/best_model.pt'
OUT = Path('results/v7')
KW = dict(topk=128, no_gating=True)
LRS = [5e-4, 2.5e-4, 1e-4]
MAX_STEP = 300
EVAL_EVERY = 10


def load_model():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12, n_pcn_layers=12,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def eval_ppl(model, data):
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1], data['y'][i:i+1]
        logits = model(x, **KW)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def freeze_backbone(model):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def main():
    print('===== step50 达峰/回落机制分解（纯 CPU, fp32）=====')
    print(f'torch threads: {torch.get_num_threads()}\n')

    # 数据：协议对齐 phase3_edge（TS 3 用户×6序列 + 2 条同分布测试）
    torch.manual_seed(0)
    train_loader, _, _ = get_dataloaders(seq_len=256, batch_size=32,
                                         num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        batches.append((x, y))
    users, tests = [], []
    for u in range(3):
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
        tests.append({'x': batches[u*2+1][0][:2], 'y': batches[u*2+1][1][:2]})

    # 通用探针：WT validation 前 4 块（预训练域，检测 H3 遗忘）
    wt = WikiTextDataset(split='validation', seq_len=256)
    gen = {'x': torch.from_numpy(wt.data[:4, :-1].astype('int64')),
           'y': torch.from_numpy(wt.data[:4, 1:].astype('int64'))}

    results = {'lrs': LRS, 'max_step': MAX_STEP, 'eval_every': EVAL_EVERY, 'runs': []}
    OUT.mkdir(parents=True, exist_ok=True)

    for lr in LRS:
        for u in range(3):
            m = load_model()
            before = eval_ppl(m, tests[u])
            trainable = freeze_backbone(m)
            pre = [p.detach().clone() for p in trainable]
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
            xs, ys = users[u]['x'], users[u]['y']
            curve = []
            m.train()
            t0 = time.time()
            for step in range(1, MAX_STEP + 1):
                ci = (step - 1) % xs.size(0)
                logits = m(xs[ci:ci+1], **KW)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), ys[ci:ci+1].reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                if step % EVAL_EVERY == 0:
                    m.eval()
                    tr = eval_ppl(m, users[u])
                    te = eval_ppl(m, tests[u])
                    gp = eval_ppl(m, gen)
                    dw = sum((p.detach() - p0).norm().item()
                             for p, p0 in zip(trainable, pre)) \
                        / sum(p0.norm().item() for p0 in pre)
                    curve.append({'step': step, 'time_s': round(time.time() - t0, 1),
                                  'train_ppl': round(tr, 2), 'test_ppl': round(te, 2),
                                  'gen_ppl': round(gp, 2),
                                  'rel_dw': round(dw, 5),
                                  'imp_pct': round((1 - te / before) * 100, 1)})
                    m.train()
            rec = {'lr': lr, 'user': u + 1, 'test_before': round(before, 2),
                   'curve': curve}
            results['runs'].append(rec)
            json.dump(results, open(OUT / 'step50_mechanism.json', 'w'),
                      indent=2, ensure_ascii=False)  # 增量保存

            peak = max(curve, key=lambda c: c['imp_pct'])
            c300 = curve[-1]
            print(f'  lr={lr:g} user{u+1}: before={before:.0f} '
                  f'峰值 step{peak["step"]} ({peak["imp_pct"]:+.1f}%, '
                  f'train {peak["train_ppl"]:.1f}) -> step300 '
                  f'({c300["imp_pct"]:+.1f}%, train {c300["train_ppl"]:.1f}) '
                  f'gen {gen["x"].size(0)}seq: {curve[0]["gen_ppl"]:.1f}'
                  f'->{c300["gen_ppl"]:.1f}')
            del m, pre

    print(f'\n输出: {OUT / "step50_mechanism.json"}')


if __name__ == '__main__':
    main()
