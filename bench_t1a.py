#!/usr/bin/env python3
"""T1a — 门控 bmm 重构吞吐基准（训练步口径，对照 bench_compile 的历史数字）"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
VOCAB, B, N = 50257, 32, 256
WARMUP, STEPS = 5, 20


def bench(model, fwd, label):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda')
    x = torch.randint(0, VOCAB, (B, N), device=DEV)
    y = x.clone()

    def step():
        opt.zero_grad()
        with torch.autocast('cuda'):
            loss = F.cross_entropy(fwd(model, x).reshape(-1, VOCAB), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    tps = B * N * STEPS / (time.time() - t0)
    print(f'  {label:<34} {tps:>9,.0f} tok/s   peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    return tps


torch.manual_seed(0)
tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                      ffn_dim=1024, dropout=0.0, max_seq_len=256, attn_impl='sdpa').to(DEV)
base = bench(tf, lambda m, x: m(x), 'Transformer (SDPA)')

torch.manual_seed(0)
pcn_bmm = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                   d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV)
t_bmm = bench(pcn_bmm, lambda m, x: m(x, topk=64), 'PCN 门控 bmm（重构后）')

pcn_loop = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                    d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV)
for blk in pcn_loop.layers:
    blk.pcn_layer.use_bmm_gate = False
t_loop = bench(pcn_loop, lambda m, x: m(x, topk=64), 'PCN 门控 loop（原实现）')

print(f'\n  bmm/loop 提速: {t_bmm/t_loop:.2f}x   |   bmm 相对 TF: {t_bmm/base*100:.0f}%（判据 ≥2x / 目标 80%）')
import json
json.dump({'tf': base, 'pcn_bmm': t_bmm, 'pcn_loop': t_loop,
           'speedup': round(t_bmm/t_loop, 2), 'pct_of_tf': round(t_bmm/base*100, 1)},
          open('results/efficiency/t1a_bench.json', 'w'), indent=2)
print('saved -> results/efficiency/t1a_bench.json')
