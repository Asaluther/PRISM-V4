#!/usr/bin/env python3
"""T4 — 能耗基准（GOAL V5#5，首次执行）：NVML 功率采样 → J/token

口径：
  训练：autocast+AdamW 步进循环（与 t1a 同协议），B=32 N=256
  推理：纯前向（与 t3 同协议），batch 1（终端）与 32（服务端）
架构：TF(SDPA) / PCN-no_gating（旗舰变体）/ PCN-门控bmm（效率线）

输出：tok/s、平均功率、绝对 J/token 与扣空闲净 J/token；
并与既有 tokens-to-PPL 数据组合，给出「TS 5K 步(40.96M tokens)训到目标 PPL 的能耗」示例。
注意：4080 是台式独显口径，绝对值不外推到手机 SoC；结论看架构间比值。

用法：.venv/Scripts/python.exe bench_energy.py
"""
import sys, time, json, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pynvml
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
VOCAB, B, N = 50257, 32, 256
OUT = Path('results/efficiency')

TRAIN_STEPS, INFER_STEPS = 60, 300
SAMPLE_MS = 50


class PowerSampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def read_once(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0

    def start(self):
        self.samples = []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.samples.append(self.read_once())
                self._stop.wait(SAMPLE_MS / 1000)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.samples


def cooldown(sampler, idle_w, timeout_s=15):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(1.0)
        if sampler.read_once() < idle_w + 5:
            return
    time.sleep(2.0)


def measure(sampler, idle_w, workload, label):
    """workload() 自带 warmup+sync，返回 (tokens, seconds)"""
    tokens, seconds = workload()
    samples = sampler.stop()
    mean_w = sum(samples) / len(samples)
    j_abs = mean_w * seconds
    j_net = (mean_w - idle_w) * seconds
    row = {'tok_per_s': round(tokens / seconds),
           'mean_w': round(mean_w, 1), 'idle_w': round(idle_w, 1),
           'n_samples': len(samples), 'duration_s': round(seconds, 2),
           'j_per_tok_abs': round(j_abs / tokens, 6),
           'j_per_tok_net': round(j_net / tokens, 6)}
    print(f'  {label:<36} {row["tok_per_s"]:>9,} tok/s  {row["mean_w"]:>6.1f} W  '
          f'{row["j_per_tok_abs"]*1e3:>7.3f} mJ/tok (abs)')
    return row


def train_workload(model, fwd, sampler):
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

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    sampler.start()

    def timed():
        t0 = time.time()
        for _ in range(TRAIN_STEPS):
            step()
        torch.cuda.synchronize()
        return B * N * TRAIN_STEPS, time.time() - t0
    return timed


def infer_workload(model, fwd, b, sampler):
    x = torch.randint(0, VOCAB, (b, N), device=DEV)

    def run(n):
        for _ in range(n):
            fwd(model, x)
        torch.cuda.synchronize()

    with torch.no_grad():
        run(10)
        sampler.start()

        def timed():
            t0 = time.time()
            run(INFER_STEPS)
            return b * N * INFER_STEPS, time.time() - t0
        return timed


def build_models():
    torch.manual_seed(0)
    tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          ffn_dim=1024, dropout=0.0, max_seq_len=256,
                          attn_impl='sdpa').to(DEV)
    ng = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                  d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV)
    gate = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                    d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed').to(DEV)
    return {'tf': (tf, lambda m, x: m(x)),
            'no_gating': (ng, lambda m, x: m(x, topk=64, no_gating=True)),
            'gate_bmm': (gate, lambda m, x: m(x, topk=64))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sampler = PowerSampler()
    time.sleep(2.0)
    idle_w = min(sampler.read_once() for _ in range(10))
    print(f'===== T4 能耗基准 =====\nGPU 空闲功率: {idle_w:.1f} W\n')

    models = build_models()
    results = {'gpu': 'RTX 4080 16GB', 'idle_w': round(idle_w, 1),
               'protocol': {'train': f'autocast+AdamW B={B} N={N} {TRAIN_STEPS}步',
                            'infer': f'fp32 前向 N={N} {INFER_STEPS}步',
                            'sample_ms': SAMPLE_MS},
               'train': {}, 'infer': {}}

    print(f'--- 训练口径 (B={B}) ---')
    for kind, (m, fwd) in models.items():
        m.train()
        wl = train_workload(m, fwd, sampler)
        results['train'][kind] = measure(sampler, idle_w, wl, f'{kind} 训练')
        cooldown(sampler, idle_w)

    print('\n--- 推理口径 ---')
    for kind, (m, fwd) in models.items():
        m.eval()
        for b in (1, 32):
            with torch.no_grad():
                wl = infer_workload(m, fwd, b, sampler)
            results['infer'][f'{kind}_b{b}'] = measure(sampler, idle_w, wl,
                                                       f'{kind} 推理 b={b}')
            cooldown(sampler, idle_w)

    # 组合示例：TS 5K 步预算（40.96M tokens）训到各自最优 PPL 的净能耗
    # tokens-to-PPL 取自 fairness_final.json 的 4-seed 均值
    print('\n--- 组合：TS 40.96M-token 预算训到 fairness PPL 的净能耗 ---')
    combo = {}
    ppl_ref = {'no_gating': 16.50, 'tf': 15.08}  # 4-seed 均值（fairness_final）
    tokens = 5000 * 32 * 256
    for kind in ('tf', 'no_gating'):
        jpt = results['train'][kind]['j_per_tok_net']
        kj = jpt * tokens / 1e3
        combo[kind] = {'ppl_ref': ppl_ref[kind], 'tokens': tokens,
                       'net_kJ': round(kj, 1)}
        if kind in ppl_ref:
            print(f'  {kind:<10} PPL≈{ppl_ref[kind]:.1f}  净能耗 {kj:,.0f} kJ')
    results['to_ppl_ts5k'] = combo

    json.dump(results, open(OUT / 't4_energy.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/t4_energy.json')


if __name__ == '__main__':
    main()
