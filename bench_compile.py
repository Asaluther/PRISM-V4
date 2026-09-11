#!/usr/bin/env python3
"""PRISM V5 — V5.2.1：门控加速基准（eager vs torch.compile vs Transformer）

问题：PCN 门控的 Python 分块循环实现吞吐仅为 SDPA attention 的 ~29%。
目标：torch.compile 能否把门控（含 Top-K 稀疏化）推到 attention 吞吐的 ≥80%。

测量：训练步（前向+反向+优化器）tokens/s，batch 32 × seq 256 × 12L-d256。
twopass 也测（计算量 ×2，compile 对两遍循环的收益是关键）。
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
VOCAB = 50257
B, N = 32, 256
WARMUP, STEPS = 5, 20


def make_batch():
    ids = torch.randint(0, VOCAB, (B, N), device=DEV)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def bench(model, forward_fn, label):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x, y = make_batch()
    scaler = torch.amp.GradScaler('cuda')

    def step():
        opt.zero_grad()
        with torch.autocast('cuda'):
            logits = forward_fn(model, x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(WARMUP):  # 含 compile 编译时间（不计时）
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tps = B * N * STEPS / dt
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:<42} {tps:>10,.0f} tok/s  ({dt/STEPS*1000:.0f} ms/step, peak {vram:.2f} GB)")
    torch.cuda.reset_peak_memory_stats()
    return tps


def fwd_eager(m, x):
    return m(x, topk=64)


def fwd_eager_2p(m, x):
    return m(x, topk=64, feedback_mode='two_pass')


def main():
    out = Path('results/analysis_v2')
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"GPU: {torch.cuda.get_device_name(0)} | {B}×{N} 训练步（fwd+bwd+opt）\n")

    # Transformer 基线
    tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          ffn_dim=1024, dropout=0.0, max_seq_len=512).to(DEV)
    results['transformer'] = bench(tf, lambda m, x: m(x), "Transformer (SDPA attention)")

    # PCN eager
    pcn = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                   d_gate=64, dropout=0.0, max_seq_len=512, init_mode='fixed').to(DEV)
    results['pcn_eager'] = bench(pcn, fwd_eager, "PCN eager (gate d64/tk64)")
    results['pcn_eager_twopass'] = bench(pcn, fwd_eager_2p, "PCN eager two_pass")

    # PCN compile
    torch.cuda.reset_peak_memory_stats()
    cmodel = torch.compile(PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                    n_heads=4, d_gate=64, dropout=0.0,
                                    max_seq_len=512, init_mode='fixed').to(DEV))
    try:
        results['pcn_compile'] = bench(cmodel, lambda m, x: m(x, topk=64),
                                       "PCN torch.compile")
        results['pcn_compile_twopass'] = bench(cmodel, lambda m, x: m(x, topk=64, feedback_mode='two_pass'),
                                               "PCN torch.compile two_pass")
    except Exception as e:
        print(f"  PCN torch.compile 失败: {type(e).__name__}: {str(e)[:200]}")
        results['pcn_compile_error'] = str(e)[:500]

    # 汇总
    base = results['transformer']
    print("\n===== 相对 Transformer 吞吐 =====")
    for k in ('pcn_eager', 'pcn_compile', 'pcn_eager_twopass', 'pcn_compile_twopass'):
        if isinstance(results.get(k), (int, float)):
            print(f"  {k:<24} {results[k]/base*100:>6.1f}%")
    results['_meta'] = {'gpu': torch.cuda.get_device_name(0), 'B': B, 'N': N,
                        'steps': STEPS, 'note': '训练步吞吐，含优化器'}
    with open(out / 'throughput_compile.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n输出: {out}/throughput_compile.json")


if __name__ == '__main__':
    main()
