#!/usr/bin/env python3
"""phase3_edge — 混合架构 96M 端侧链路验证（纯 CPU）

测量一：CPU 推理基准（协议对齐 V7 b2_cpu_bench：预热 3 + 计时 10，B×N 网格，
合成随机 token，自回归生成 + 样本文本）
测量二：CPU 冷启动步数扫描（协议对齐 V7 b3_cpu_coldstart：TS 3 用户 × 6 序列，
AdamW 5e-4，冻结全部 TF 骨干 + emb，fp32；在 {50,100,150,200,300} 步各评估一次
——把「30 秒判据可能超」转化为时间-改善权衡曲线）

预注册判据：① b1_s256 ≥ 500 tok/s  ② 生成 ≥ 10 tok/s
③a 300 步改善 > 0  ③b 达最终改善一半 ≤ 60s  附：全程无网络调用

用法：.venv/Scripts/python.exe phase3_edge.py
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

VOCAB = 50257
CKPT = 'results/wt_094m_hyb_s0/best_model.pt'
OUT = Path('results/v7')
KW = dict(topk=128, no_gating=True)   # 其余前向参量均为 HybridModel 默认值


def load_model():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12, n_pcn_layers=12,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def bench(model, B, N, warmup=3, steps=10):
    x = torch.randint(0, VOCAB, (B, N))
    for _ in range(warmup):
        model(x, **KW)
    t0 = time.time()
    for _ in range(steps):
        model(x, **KW)
    dt = time.time() - t0
    return B * N * steps / dt, dt / steps * 1000


@torch.no_grad()
def bench_gen(model, max_new=20):
    ids = torch.randint(0, VOCAB, (1, 20))
    for _ in range(1):                      # 预热
        model(ids[:, -256:], **KW)
    t0 = time.time()
    for _ in range(max_new):
        logits = model(ids[:, -256:], **KW)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    dt = time.time() - t0
    return max_new / dt, dt / max_new * 1000, ids


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
    print('===== 阶段③ 混合架构 96M 端侧链路（纯 CPU）=====')
    print(f'torch threads: {torch.get_num_threads()}\n')

    m = load_model()
    n_params = m.count_params()
    print(f'模型: {n_params/1e6:.1f}M 参数, fp32 ~{n_params*4/1024/1024:.0f} MB '
          f'(ckpt {Path(CKPT).stat().st_size/1024/1024:.0f} MB)')

    # ---- 测量一：推理基准 ----
    print('\n--- CPU 推理基准 ---')
    results = {'model_params_M': round(n_params / 1e6, 1),
               'model_mb_fp32': round(n_params * 4 / 1024 / 1024, 1),
               'threads': torch.get_num_threads()}
    infer = {}
    for B in (1, 2, 4):
        for N in (128, 256):
            tps, lat = bench(m, B, N)
            infer[f'b{B}_s{N}'] = {'tok_per_s': round(tps, 1),
                                   'latency_ms': round(lat, 1)}
            print(f'  B={B} N={N}: {tps:8.1f} tok/s  ({lat:.1f} ms/batch)')
    gtps, gms, ids = bench_gen(m)
    infer['generation'] = {'tok_per_s': round(gtps, 1), 'ms_per_tok': round(gms, 1)}
    print(f'  生成: {gtps:.1f} tok/s ({gms:.1f} ms/tok)')
    results['inference'] = infer

    # 样本文本（域 sanity——WT 预训练，应输出维基风格碎片）
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(
        str(Path(__file__).parent / 'tokenizer'))
    sample = tok.decode(ids[0, 20:].tolist())
    print(f'  样本: {sample[:180]!r}')

    # ---- 测量二：冷启动步数扫描 ----
    print('\n--- CPU 冷启动（3 用户 × 6 序列，冻结全部 TF 骨干）---')
    train_loader, _, _ = get_dataloaders(seq_len=256, batch_size=32,
                                         num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        batches.append((x, y))
    users = []
    for u in range(3):
        xs = [batches[u*2][0][i:i+1] for i in range(6)]
        ys = [batches[u*2][1][i:i+1] for i in range(6)]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})
    tests = []
    for u in range(3):
        tests.append({'x': batches[u*2+1][0][:2], 'y': batches[u*2+1][1][:2]})

    CHECKPOINTS = [50, 100, 150, 200, 300]
    cold = []
    for u in range(3):
        m = load_model()
        before = eval_ppl(m, tests[u])
        trainable = freeze_backbone(m)
        opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
        xs, ys = users[u]['x'], users[u]['y']
        m.train()
        curve = []
        t0 = time.time()
        for step in range(1, 301):
            ci = (step - 1) % xs.size(0)
            x, y = xs[ci:ci+1], ys[ci:ci+1]
            logits = m(x, **KW)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if step in CHECKPOINTS:
                m.eval()
                pa = eval_ppl(m, tests[u])
                curve.append({'step': step, 'time_s': round(time.time() - t0, 1),
                              'ppl': round(pa, 1),
                              'improvement_pct': round((1 - pa / before) * 100, 1)})
                m.train()
        total_t = time.time() - t0
        cold.append({'user': u + 1, 'before': round(before, 1),
                     'curve': curve, 'time_s_300': round(total_t, 1)})
        fin = curve[-1]['improvement_pct']
        half = next((c for c in curve
                     if c['improvement_pct'] >= fin / 2), curve[-1])
        print(f'  user{u+1}: {before:.0f} -> {curve[-1]["ppl"]:.0f} '
              f'({fin:+.1f}%, 300 步 {total_t:.0f}s; 半程改善于 '
              f'step{half["step"]}/{half["time_s"]:.0f}s)')
        del m
    results['coldstart'] = cold

    # ---- 判据 ----
    print('\n===== 预注册判据 =====')
    tps256 = infer['b1_s256']['tok_per_s']
    print(f'  ① b1_s256 ≥500 tok/s: {tps256} → {"✅" if tps256 >= 500 else "❌"}')
    print(f'  ② 生成 ≥10 tok/s: {gtps:.1f} → {"✅" if gtps >= 10 else "❌"}')
    fins = [c['curve'][-1]['improvement_pct'] for c in cold]
    avg_fin = sum(fins) / len(fins)
    print(f'  ③a 300 步平均改善 >0: {avg_fin:+.1f}% → {"✅" if avg_fin > 0 else "❌"}')
    half_times = []
    for c in cold:
        fin = c['curve'][-1]['improvement_pct']
        half = next((x for x in c['curve'] if x['improvement_pct'] >= fin / 2),
                    c['curve'][-1])
        half_times.append(half['time_s'])
    avg_half = sum(half_times) / len(half_times)
    print(f'  ③b 半程改善 ≤60s: {avg_half:.0f}s → {"✅" if avg_half <= 60 else "❌"}')

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'phase3_edge_hyb.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "phase3_edge_hyb.json"}')


if __name__ == '__main__':
    main()
