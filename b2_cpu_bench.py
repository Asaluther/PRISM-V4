#!/usr/bin/env python3
"""B2 — PC 端 CPU 推理基准（笔记本终端口径）"""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.pcn import PCNModel

VOCAB = 50257
CKPT = 'results/wt_causal_no_gating_s0/best_model.pt'
OUT = Path('results/v7')


def load_model():
    m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                 d_gate=64, dropout=0.0, max_seq_len=256,
                 init_mode='fixed', act_sparse=0.0)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def bench(model, B, N, steps=10, warmup=3):
    x = torch.randint(0, VOCAB, (B, N))
    for _ in range(warmup):
        model(x, topk=64, no_gating=True)
    t0 = time.time()
    for _ in range(steps):
        model(x, topk=64, no_gating=True)
    dt = time.time() - t0
    return round(B * N * steps / dt), round(dt / steps * 1000, 1)


@torch.no_grad()
def bench_gen(model, max_new=30):
    ids = torch.randint(0, VOCAB, (1, 20))
    # warmup
    model(ids, topk=64, no_gating=True)
    t0 = time.time()
    for _ in range(max_new):
        logits = model(ids[:, -256:], topk=64, no_gating=True)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    dt = time.time() - t0
    return round(max_new / dt, 1), round(dt / max_new * 1000, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'===== B2: CPU 推理基准 =====')
    print(f'Threads: {torch.get_num_threads()}\n')

    m = load_model()
    n_params = sum(p.numel() for p in m.parameters())
    model_mb = n_params * 4 / 1024 / 1024  # fp32
    print(f'模型: {n_params/1e6:.1f}M 参数, ~{model_mb:.0f} MB (fp32)\n')

    results = {'model_params_M': round(n_params/1e6, 1), 'model_mb_fp32': round(model_mb)}

    print(f'{"batch":>6} {"seq":>5} {"tok/s":>10} {"latency(ms)":>12}')
    for B in (1, 2, 4):
        for N in (128, 256):
            tps, lat = bench(m, B, N)
            results[f'b{B}_s{N}'] = {'tok_per_s': tps, 'latency_ms': lat}
            print(f'{B:>6} {N:>5} {tps:>10,} {lat:>12}')

    print('\n--- 自回归生成（终端交互口径）---')
    gen_tps, ms = bench_gen(m, max_new=20)
    results['generation'] = {'tok_per_s': gen_tps, 'ms_per_tok': ms}
    print(f'  {gen_tps} tok/s  ({ms} ms/token)')
    print(f'  50 token 回复耗时: {50/gen_tps:.1f}s')

    b1 = results['b1_s256']['tok_per_s']
    print(f'\n判定: batch1 seq256 {b1:,} tok/s（目标≥50）{"PASS" if b1 >= 50 else "FAIL"}')

    json.dump(results, open(OUT / 'b2_cpu_infer.json', 'w'), indent=2)
    print(f'saved -> {OUT}/b2_cpu_infer.json')


if __name__ == '__main__':
    main()
