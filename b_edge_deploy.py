#!/usr/bin/env python3
"""B1+B2 — 模型导出 + PC 端 CPU 推理基准

B1: 将稠密 PCN checkpoint 导出为 TorchScript（脱离训练环境可部署格式）
B2: 在 CPU（无 GPU）上测推理吞吐/延迟/内存——笔记本终端口径

用法：CUDA_VISIBLE_DEVICES=-1 python b_edge_deploy.py  # 强制 CPU
"""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
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


def export_torchscript(model, path):
    """导出 TorchScript 模型"""
    # 用 tracing 导出（固定 topk=64, no_gating=True 路径）
    example = torch.randint(0, VOCAB, (1, 128))

    class PCNForExport(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids):
            return self.m(ids, topk=64, no_gating=True)

    wrapper = PCNForExport(model).eval()
    scripted = torch.jit.trace(wrapper, example)
    scripted.save(str(path))
    return scripted


@torch.no_grad()
def bench_cpu(model, B, N, steps=20, warmup=5):
    """CPU 推理基准"""
    x = torch.randint(0, VOCAB, (B, N))
    for _ in range(warmup):
        model(x)
    t0 = time.time()
    for _ in range(steps):
        model(x)
    dt = time.time() - t0
    tps = B * N * steps / dt
    latency_first = dt / steps * 1000  # ms per batch
    return round(tps), round(latency_first, 1)


@torch.no_grad()
def bench_generation(model, tok, prompt_ids, max_new=50):
    """自回归生成延迟（终端最关心的指标）"""
    ids = prompt_ids.clone()
    t0 = time.time()
    for _ in range(max_new):
        logits = model(ids[:, -256:])
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    dt = time.time() - t0
    tok_per_sec = max_new / dt
    ms_per_tok = dt / max_new * 1000
    return round(tok_per_sec, 1), round(ms_per_tok, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cpu')  # 强制 CPU
    print(f'===== B1+B2: 模型导出 + CPU 推理基准 =====')
    print(f'Device: {dev} ({torch.get_num_threads()} threads)\n')

    # B1: 导出
    print('--- B1: TorchScript 导出 ---')
    model = load_model()
    ts_path = OUT / 'pcn_edge.ts'
    scripted = export_torchscript(model, ts_path)
    size_mb = ts_path.stat().st_size / 1024 / 1024
    print(f'  TorchScript: {ts_path.name} ({size_mb:.1f} MB)')

    # fp16 转换（减半存储）
    model_fp16 = load_model().half()
    ts_fp16 = OUT / 'pcn_edge_fp16.ts'
    s16 = export_torchscript(model_fp16, ts_fp16)
    size16 = ts_fp16.stat().st_size / 1024 / 1024
    print(f'  TorchScript fp16: {ts_fp16.name} ({size16:.1f} MB)')

    # B2: CPU 推理基准（用原始模型，不用 TorchScript——兼容性更好）
    print('\n--- B2: CPU 推理基准 ---')
    m = load_model()

    results = {}
    print(f'{"batch":>6} {"seq":>5} {"tok/s":>10} {"latency(ms)":>12}')
    for B in (1, 2, 4):
        for N in (128, 256):
            tps, lat = bench_cpu(m, B, N)
            results[f'b{B}_s{N}'] = {'tok_per_s': tps, 'latency_ms': lat}
            print(f'{B:>6} {N:>5} {tps:>10,} {lat:>12}')

    # 自回归生成延迟（batch 1，模拟终端交互）
    from transformers import GPT2TokenizerFast
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    prompt = tok.encode("Once upon a time")
    prompt_ids = torch.tensor([prompt])

    print('\n--- 自回归生成（终端交互口径）---')
    gen_tps, ms_per_tok = bench_generation(m, tok, prompt_ids, max_new=30)
    print(f'  生成速度: {gen_tps} tok/s  ({ms_per_tok} ms/token)')
    print(f'  30 token 生成耗时: {30/gen_tps:.1f}s')
    results['generation'] = {'tok_per_s': gen_tps, 'ms_per_tok': ms_per_tok}

    # 内存占用
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux; on Windows may differ
    print(f'\n  峰值内存: ~{mem_mb:.0f} MB')
    results['memory_mb'] = round(mem_mb)

    # 判定
    b1_s256 = results['b1_s256']['tok_per_s']
    print(f'\n===== 判定 =====')
    print(f'  batch1 seq256 吞吐: {b1_s256:,} tok/s（目标 ≥50）{"✅" if b1_s256 >= 50 else "❌"}')
    print(f'  生成延迟: {ms_per_tok} ms/tok（目标 <100ms）{"✅" if ms_per_tok < 100 else "❌"}')
    print(f'  模型大小: {size_mb:.1f} MB fp32 / {size16:.1f} MB fp16')

    json.dump(results, open(OUT / 'b2_cpu_infer.json', 'w'), indent=2)
    print(f'\nsaved -> {OUT}/b2_cpu_infer.json + pcn_edge.ts')


if __name__ == '__main__':
    main()
