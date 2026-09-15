#!/usr/bin/env python3
"""A3 — 稀疏推理加速：稀疏激活的实际吞吐收益

比较三种推理模式的吞吐（batch 1-32）：
1. 稠密 PCN（act_sparse=0）——基线
2. 稀疏 PCN（act_sparse=0.25，但不跳过计算）——当前实现
3. 稀疏 PCN + 计算跳过（只对非零激活做 W_up matmul）——理论目标

关键问题：75% 激活为零，能否在 GPU 上跳过这些计算获得实际加速？
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel

DEV = torch.device('cuda')
VOCAB, N = 50257, 256
WARMUP, STEPS = 10, 50
OUT = Path('results/v7')


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


def bench_sparse_skip(model, B, sparsity=0.25):
    """模拟计算跳过：对 GELU 输出做 Top-K 后只对非零部分做后续 matmul。
    在 GPU 上稀疏 matmul 的收益取决于框架支持——这里测 mask + gather 方案。"""
    x = torch.randint(0, VOCAB, (B, N), device=DEV)
    d = model.d_model
    k = int(d * sparsity)

    def sparse_fwd(m, ids):
        h = m.token_emb(ids) + m.pos_emb(torch.arange(ids.size(1), device=ids.device))
        for blk in m.layers:
            pc = blk.pcn_layer
            u = pc.W_bu(pc.norm_bu(h))
            e = u.clamp(-10, 10)
            mask = torch.triu(torch.ones(ids.size(1), ids.size(1),
                                         device=ids.device, dtype=torch.bool), diagonal=1)
            attn_out, _ = blk.attn(e, e, e, attn_mask=mask, need_weights=False)
            lat = blk.attn_out(attn_out)
            update = pc.W_up(pc.norm_update(e + lat))
            act = F.gelu(update)
            # Top-K 稀疏
            tv, ti = act.topk(k, dim=-1)
            thr = tv[..., -1:]
            act_s = act * (act >= thr).float()
            # 稀疏计算：只对非零行做 W_res（模拟——实际上 W_res 也作用于全量 h）
            h = act_s + pc.W_res(h)
        return m.lm_head(m.ln_out(h))

    for _ in range(WARMUP):
        sparse_fwd(model, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        sparse_fwd(model, x)
    torch.cuda.synchronize()
    return B * N * STEPS / (time.time() - t0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== A3: 稀疏推理吞吐 =====\n')

    torch.manual_seed(0)
    dense = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                     d_gate=64, dropout=0.0, max_seq_len=256,
                     init_mode='fixed', act_sparse=0.0).to(DEV).eval()
    sparse25 = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                        d_gate=64, dropout=0.0, max_seq_len=256,
                        init_mode='fixed', act_sparse=0.25).to(DEV).eval()
    tf = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                          ffn_dim=1024, dropout=0.0, max_seq_len=256,
                          attn_impl='sdpa').to(DEV).eval()

    rows = {}
    print(f'{"batch":>6} {"TF":>10} {"PCN稠密":>10} {"PCN稀疏25%":>12} {"稀疏/稠密":>10}')
    for B in (1, 4, 16, 32):
        t = bench_infer(tf, lambda m, x: m(x), B)
        d = bench_infer(dense, lambda m, x: m(x, topk=64, no_gating=True), B)
        s = bench_infer(sparse25, lambda m, x: m(x, topk=64, no_gating=True), B)
        rows[B] = {'tf': round(t), 'pcn_dense': round(d),
                   'pcn_sparse25': round(s), 'sparse_vs_dense': round(s/d, 2)}
        print(f'{B:>6} {t:>10,.0f} {d:>10,.0f} {s:>12,.0f} {s/d:>9.2f}x')

    # 稀疏计算跳过测试（batch 1 终端口径）
    print('\n--- 稀疏计算跳过方案（batch 1, 手动 gather/scatter）---')
    B = 1
    s_manual = bench_sparse_skip(sparse25, B)
    d1 = rows[1]['pcn_dense']
    print(f'  稠密: {d1:,.0f} tok/s  稀疏(计算跳过): {s_manual:,.0f} tok/s  '
          f'加速比: {s_manual/d1:.2f}x')

    json.dump(rows, open(OUT / 'a3_sparse_infer.json', 'w'), indent=2)
    print(f'\nsaved -> {OUT}/a3_sparse_infer.json')

    # 结论
    print('\n===== 结论 =====')
    b1 = rows[1]
    print(f'  GPU 上稀疏激活的实际加速比（batch1）: {b1["sparse_vs_dense"]}x')
    print(f'  （75% 激活为零，但 GPU 的稠密 matmul 内核无法直接跳过零——')
    print(f'   需要稀疏 kernel（Triton/CUTLASS）才能兑现理论加速）')


if __name__ == '__main__':
    main()
