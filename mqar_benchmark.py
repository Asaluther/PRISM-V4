#!/usr/bin/env python3
"""MQAR/recall 正式基准（GOAL V5#4 未竟项）——把 PCN 放上「回忆-效率」前沿

任务：标准 Multi-Query Associative Recall（Based/Zoology 口径）——
  序列 = [k1 v1 k2 v2 ... kn vn | SEP | q1 q2 ... q16]，键值 token 不相交，
  查询键重复出现于键区；查询位下一 token = 对应值。
  难度 = 字典规模 n（记忆跨度 2n token）。
指标：查询位 argmax 准确率（≥0.9 = 解出）。

臂（真实项目模型，玩具规模 D=64/L=12，各自 LM 最优 lr——教训 8 公平纪律）：
  TF@1e-3(sdpa) / TF-decay@1e-3(SSM 固定衰减核代理) / PCN no_gating@3e-4 / PCN 门控@3e-4
出线：n ∈ {16,32,48} × seed {0,1}；2000 步。
效率轴：四臂 N=256 b32 推理吞吐（同脚本测量），与准确率合成权衡表。

用法：.venv/Scripts/python.exe mqar_benchmark.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
OUT = Path('results/mechanism')
K_VOCAB, V_VOCAB, SEP = 64, 64, 128  # 键 0-63，值 64-127，SEP=128 → vocab=129
VOCAB = 129
Q = 16
NS = [16, 32, 48]
SEEDS = [0, 1]
STEPS, EVAL_EVERY = 2000, 250
D, L, HEADS = 64, 12, 4


def gen_mqar(B, n, seed=None):
    """seq = [k1 v1 ... kn vn | SEP | q1 ... qQ | END]；x/y 为错位切分。
    查询位 q_i 在 x 的索引 2n+1+i；这些位的预测目标覆盖为对应值（标准 MQAR
    口径：查询位的下一 token 分布训练为值，实际后续 token 是下一查询）。"""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    if g is not None:
        keys = torch.stack([torch.randperm(K_VOCAB, generator=g)[:n] for _ in range(B)])
        qidx = torch.stack([torch.randperm(n, generator=g)[:Q] for _ in range(B)])
        vals = torch.randint(0, V_VOCAB, (B, n), generator=g)
    else:
        keys = torch.stack([torch.randperm(K_VOCAB)[:n] for _ in range(B)])
        qidx = torch.stack([torch.randperm(n)[:Q] for _ in range(B)])
        vals = torch.randint(0, V_VOCAB, (B, n))
    mem = torch.empty(B, 2 * n, dtype=torch.long)
    mem[:, 0::2] = keys
    mem[:, 1::2] = vals + K_VOCAB
    queries = torch.gather(keys, 1, qidx)
    qvals = torch.gather(vals, 1, qidx)
    seq = torch.cat([mem, torch.full((B, 1), SEP, dtype=torch.long),
                     queries, torch.full((B, 1), SEP, dtype=torch.long)], dim=1)
    x = seq[:, :-1]
    y = seq[:, 1:].clone()
    qm = torch.zeros_like(y, dtype=torch.bool)
    for i in range(Q):
        y[:, 2 * n + 1 + i] = qvals[:, i] + K_VOCAB
        qm[:, 2 * n + 1 + i] = True
    return x.to(DEV), y.to(DEV), qm.to(DEV)


def build(arm):
    if arm == 'tf':
        return TransformerModel(vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=HEADS,
                                ffn_dim=4 * D, dropout=0.0, max_seq_len=256, attn_impl='sdpa')
    if arm == 'decay':
        return TransformerModel(vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=HEADS,
                                ffn_dim=4 * D, dropout=0.0, max_seq_len=256, attn_impl='decay')
    return PCNModel(vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=HEADS,
                    d_gate=16, dropout=0.0, max_seq_len=256, init_mode='fixed')


def fwd(arm, m, x):
    if arm == 'ng':
        return m(x, topk=16, no_gating=True)
    if arm == 'gate':
        return m(x, topk=16)
    return m(x)


@torch.no_grad()
def query_acc(model, arm, n, batches=3):
    model.eval()
    hit, tot = 0, 0
    for i in range(batches):
        x, y, qm = gen_mqar(64, n, seed=9000 + i)
        logits = fwd(arm, model, x)
        pred = logits.argmax(-1)
        mf = qm.reshape(-1)
        hit += (pred.reshape(-1)[mf] == y.reshape(-1)[mf]).sum().item()
        tot += mf.sum().item()
    model.train()
    return hit / tot


def train_one(arm, n, seed, lr, steps=STEPS, batch=32):
    torch.manual_seed(seed)
    model = build(arm).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    for step in range(1, steps + 1):
        x, y, _ = gen_mqar(batch, n)
        loss = F.cross_entropy(fwd(arm, model, x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            acc = query_acc(model, arm, n)
            curve.append({'step': step, 'acc': round(acc, 3)})
            if acc >= 0.999:
                break  # 解出即停，省预算
    return curve


@torch.no_grad()
def infer_tput(arm):
    """N=256 b32 推理吞吐（与 t3 同口径，toy 模型，同脚本自含）"""
    m = build(arm).to(DEV).eval()
    x = torch.randint(0, VOCAB, (32, 256), device=DEV)
    for _ in range(10):
        fwd(arm, m, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        fwd(arm, m, x)
    torch.cuda.synchronize()
    return 32 * 256 * 50 / (time.time() - t0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [('tf', 1e-3, 'Transformer@1e-3'),
            ('decay', 1e-3, 'TF-decay(SSM代理)@1e-3'),
            ('ng', 3e-4, 'PCN no_gating@3e-4'),
            ('gate', 3e-4, 'PCN 门控@3e-4')]
    print(f'===== MQAR 基准（D={D} L={L}, q={Q}, {STEPS}步, 解出即停）=====\n')
    results = {}
    for arm, lr, label in arms:
        results[arm] = {'label': label, 'lr': lr, 'runs': {}}
        for n in NS:
            accs = []
            for s in SEEDS:
                curve = train_one(arm, n, s, lr)
                accs.append(curve[-1]['acc'])
                results[arm]['runs'][f'n{n}_s{s}'] = curve
                print(f'  {label:<24} n={n:<2} seed{s}: 终末 acc={accs[-1]:.3f} '
                      f'({curve[-1]["step"]} 步)')
            solved = all(a >= 0.9 for a in accs)
            results[arm][f'acc_n{n}'] = accs
            results[arm][f'n{n}_solved'] = solved
            print(f'    -> n={n} {"✅ 解出" if solved else "❌"}')
        results[arm]['max_n_solved'] = max(
            [nn for nn in NS if results[arm].get(f'n{nn}_solved')] + [0])
    # 效率轴
    print('\n----- 推理吞吐（N=256 b32, toy 模型）-----')
    for arm, lr, label in arms:
        t = infer_tput(arm)
        results[arm]['infer_tok_per_s_b32'] = round(t)
        print(f'  {label:<24} {t:>9,.0f} tok/s')

    json.dump(results, open(OUT / 'mqar_benchmark.json', 'w'), indent=2)

    print('\n===== 回忆-效率权衡 =====')
    print(f'{"臂":<26}{"最大字典 n":>10}{"b32 tok/s":>12}')
    for arm, lr, label in arms:
        print(f'{label:<26}{results[arm]["max_n_solved"]:>10}'
              f'{results[arm]["infer_tok_per_s_b32"]:>12,}')
    print(f'\n输出: {OUT}/mqar_benchmark.json')


if __name__ == '__main__':
    main()
