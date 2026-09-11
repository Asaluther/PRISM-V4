#!/usr/bin/env python3
"""T5 — batch1 终端口径优化：CUDA Graph 捕获/重放（EFFICIENCY_REPORT 后续项）

问题：b1 推理 gate_bmm 仅 TF 的 ~60%（kernel 启动开销主导，t3/t4），
终端每 token 能耗 2×（t4_energy）。
手段：整前向捕获为 CUDA Graph，重放消除逐 kernel 启动开销。
公平：TF 同样 graph 化（优化对优化比较，判据 = graph(gate)/graph(TF) ≥ 90%）。

口径：b1 主口径 + b4 上下文；eager vs graph 吞吐、功率、mJ/tok；
数值等价校验（graph logits vs eager logits max|Δ|）。

用法：.venv/Scripts/python.exe bench_b1_graph.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, time, json, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from bench_energy import PowerSampler

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/efficiency')
WARMUP, STEPS = 30, 300


def build_models():
    torch.manual_seed(0)
    tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          ffn_dim=1024, dropout=0.0, max_seq_len=256,
                          attn_impl='sdpa').to(DEV).eval()
    ng = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                  d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV).eval()
    gate = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                    d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV).eval()
    return {'tf': (tf, lambda m, x: m(x)),
            'no_gating': (ng, lambda m, x: m(x, topk=64, no_gating=True)),
            'gate_bmm': (gate, lambda m, x: m(x, topk=64))}


@torch.no_grad()
def bench_eager(model, fwd, x):
    for _ in range(WARMUP):
        fwd(model, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        fwd(model, x)
    torch.cuda.synchronize()
    return STEPS * x.numel() / (time.time() - t0)


@torch.no_grad()
def capture_graph(model, fwd, x):
    """标准捕获流程：side-stream 预热 → 静态 IO → 捕获。返回 (graph, static_x, static_out)"""
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = fwd(model, static_x)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fwd(model, static_x)
    return g, static_x, static_out


@torch.no_grad()
def bench_graph(g, x):
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        g.replay()
    torch.cuda.synchronize()
    return STEPS * x.numel() / (time.time() - t0)


def measure_energy(sampler, idle_w, loop):
    """loop() 为已预热的计时体；返回 (tok_per_s, mJ_per_tok_abs)"""
    sampler.start()
    tps = loop()
    samples = sampler.stop()
    mean_w = sum(samples) / len(samples)
    # loop 内已知 STEPS×tokens；由 tps 反推时长
    return tps, round(mean_w / tps, 6), round(mean_w, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sampler = PowerSampler()
    time.sleep(2.0)
    idle_w = min(sampler.read_once() for _ in range(10))
    print(f'===== T5 batch1 CUDA Graph 优化 =====\nGPU 空闲 {idle_w:.1f} W\n')

    models = build_models()
    results = {'gpu': 'RTX 4080 16GB', 'idle_w': round(idle_w, 1),
               'warmup': WARMUP, 'steps': STEPS, 'mode': 'fp32 前向', 'rows': {}}

    for b in (1, 4):
        x = torch.randint(0, VOCAB, (b, 256), device=DEV)
        for kind, (m, fwd) in models.items():
            # 数值等价校验（必须先 replay 一次——捕获本身不执行，
            # static_out 在首次重放前是未初始化的 graph 池内存）
            g, sx, sout = capture_graph(m, fwd, x)
            with torch.no_grad():
                g.replay()
                torch.cuda.synchronize()
                ref = fwd(m, x)
                torch.cuda.synchronize()
            diff = (sout - ref).abs().max().item()
            assert diff < 1e-3, f'{kind} b{b} graph 数值不等价: |Δ|={diff}'

            row = {}
            tps_e, mj_e, w_e = measure_energy(sampler, idle_w,
                                              lambda: bench_eager(m, fwd, x))
            tps_g, mj_g, w_g = measure_energy(sampler, idle_w,
                                              lambda: bench_graph(g, x))
            row.update(eager={'tok_per_s': round(tps_e), 'mean_w': w_e,
                              'mj_per_tok': mj_e},
                       graph={'tok_per_s': round(tps_g), 'mean_w': w_g,
                              'mj_per_tok': mj_g},
                       graph_speedup=round(tps_g / tps_e, 2),
                       max_logit_diff=diff)
            results['rows'][f'{kind}_b{b}'] = row
            print(f'  [{kind} b{b}] eager {tps_e:>8,.0f} tok/s ({w_e:.0f}W, {mj_e*1e3:.2f}mJ)  '
                  f'graph {tps_g:>8,.0f} tok/s ({w_g:.0f}W, {mj_g*1e3:.2f}mJ)  '
                  f'x{tps_g/tps_e:.2f}  |Δlogit|={diff:.2e}')

    # 判定：优化对优化
    print('\n===== 判定（graph vs graph）=====')
    for b in (1, 4):
        t = results['rows'][f'tf_b{b}']['graph']['tok_per_s']
        for kind in ('no_gating', 'gate_bmm'):
            g_ = results['rows'][f'{kind}_b{b}']['graph']['tok_per_s']
            print(f'  b{b} {kind:<10} {g_/t*100:5.1f}% of TF-graph'
                  f'{"  ✅ 达标(≥90%)" if g_ >= 0.9*t else "  ❌ 未达"}')

    json.dump(results, open(OUT / 't5_b1_graph.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/t5_b1_graph.json')


if __name__ == '__main__':
    main()
