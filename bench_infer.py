#!/usr/bin/env python3
"""T3 — 推理效率基准（终端口径：batch 1-32 纯前向吞吐）

终端场景 = batch 1。测三架构：Transformer(SDPA) / PCN no_gating / PCN 门控(bmm)。
与训练口径（t1a_bench）分开报告。
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
VOCAB, N = 50257, 256
WARMUP, STEPS = 10, 50


@torch.no_grad()
def bench_infer(model, fwd, B):
    x = torch.randint(0, VOCAB, (B, N), device=DEV)
    for _ in range(WARMUP):
        fwd(model, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        fwd(model, x)
    torch.cuda.synchronize()
    return B * N * STEPS / (time.time() - t0)


def main():
    torch.manual_seed(0)
    tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          ffn_dim=1024, dropout=0.0, max_seq_len=256,
                          attn_impl='sdpa').to(DEV).eval()
    ng = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                  d_gate=64, dropout=0.0, max_seq_len=256,
                  init_mode='fixed').to(DEV).eval()
    gate = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                    d_gate=64, dropout=0.0, max_seq_len=256,
                    init_mode='fixed').to(DEV).eval()

    rows = {}
    print(f'{"batch":>6} {"TF(SDPA)":>12} {"PCN-no_gating":>14} {"PCN-门控bmm":>13}')
    for B in (1, 4, 16, 32):
        t = bench_infer(tf, lambda m, x: m(x), B)
        n = bench_infer(ng, lambda m, x: m(x, topk=64, no_gating=True), B)
        g = bench_infer(gate, lambda m, x: m(x, topk=64), B)
        rows[B] = {'tf': round(t), 'no_gating': round(n), 'gate_bmm': round(g)}
        print(f'{B:>6} {t:>12,.0f} {n:>14,.0f} {g:>13,.0f}')

    Path('results/efficiency').mkdir(exist_ok=True)
    json.dump(rows, open('results/efficiency/t3_infer.json', 'w'), indent=2)
    b1 = rows[1]
    print(f'\nbatch1 终端口径: 门控bmm 为 TF 的 {b1["gate_bmm"]/b1["tf"]*100:.0f}%')
    print('saved -> results/efficiency/t3_infer.json')


if __name__ == '__main__':
    main()
