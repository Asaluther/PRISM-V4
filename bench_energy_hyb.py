#!/usr/bin/env python3
"""bench_energy_hyb — T4 能耗口径平移到 96M 混合架构（V7 主线 ⑤）

v1 的 T4 方法（NVML 50ms 采样 → measure 契约：workload 自启采样器并返回
(tokens, seconds)）首次覆盖混合架构与终端用例全景：

  推理 b1（终端口径）：h2e4 96M fp32/fp16、TF 101M、PCN 21M（V7 锚点）
  生成（自回归 40 token）：h2e4 fp32/fp16
  适应（50 步冻结骨干，终端预算）：h2e4 fp32 总焦耳

注意（继承 v1 口径）：4080 台式独显，绝对值不外推手机 SoC；结论看比值
与量级。输出：results/efficiency/t4_hyb.json
用法：.venv/Scripts/python.exe bench_energy_hyb.py   (需 GPU)
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from bench_energy import PowerSampler, cooldown, measure
from src.models.hybrid import HybridModel
from src.models.transformer import TransformerModel
from src.models.pcn import PCNModel
from src.data.tinystories import TinyStoriesDataset

VOCAB = 50257
DEV = torch.device('cuda')
KW = dict(topk=128, no_gating=True)
KW21 = dict(topk=64, no_gating=True)
OUT = Path('results/efficiency')
SAMPLER = PowerSampler()


def load_hyb(half=False):
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load('results/wt_096m_hyb_h2e-4_s0/best_model.pt',
                                 map_location='cpu', weights_only=True))
    m = m.to(DEV).eval()
    return m.half() if half else m


def load_tf():
    m = TransformerModel(vocab_size=VOCAB, d_model=512, n_layers=24,
                         n_heads=4, ffn_dim=2048, dropout=0.0,
                         max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load('results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
                                 map_location='cpu', weights_only=True))
    return m.to(DEV).eval()


def load_pcn21():
    m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                 d_gate=64, dropout=0.0, max_seq_len=256,
                 init_mode='fixed', use_bmm_gate=True)
    m.load_state_dict(torch.load('results/wt_causal_no_gating_s0/best_model.pt',
                                 map_location='cpu', weights_only=True))
    return m.to(DEV).eval()


def timed_infer(model, kw, steps=200, b=1, n=256, warmup=20):
    """v1 契约：预热 → 启动采样 → 计时 → 返回 (tokens, seconds)"""
    x = torch.randint(0, VOCAB, (b, n), device=DEV)
    for _ in range(warmup):
        model(x, **kw) if kw else model(x)
    torch.cuda.synchronize()
    SAMPLER.start()
    t0 = time.time()
    for _ in range(steps):
        model(x, **kw) if kw else model(x)
    torch.cuda.synchronize()
    return b * n * steps, time.time() - t0


def timed_gen(model, kw, max_new=40):
    @torch.no_grad()
    def once():
        ids = torch.randint(0, VOCAB, (1, 20), device=DEV)
        if next(model.parameters()).dtype == torch.half:
            ids = ids
        for _ in range(max_new):
            logits = model(ids[:, -256:], **kw)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        torch.cuda.synchronize()
    once()  # 预热
    SAMPLER.start()
    t0 = time.time()
    once()
    return max_new, time.time() - t0


def adapt50_timed(model):
    """终端预算 50 步冻结骨干适应；返回 (steps, seconds) 供 measure"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    import numpy as np
    ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    rows = np.stack([np.asarray(r) for r in ds.data[:2]])
    d = torch.from_numpy(rows.astype('int64')).to(DEV)
    x, y = d[:, :-1].contiguous(), d[:, 1:].contiguous()
    model.train()
    for _ in range(10):   # 预热（不计入）
        ci = _ % x.size(0)
        logits = model(x[ci:ci+1], **KW)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    torch.cuda.synchronize()
    model.eval()
    # 重置适应态（重新装载干净权重语义由调用方保证——此处测能耗不测质量）
    model.train()
    SAMPLER.start()
    t0 = time.time()
    for step in range(50):
        ci = step % x.size(0)
        logits = model(x[ci:ci+1], **KW)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    torch.cuda.synchronize()
    model.eval()
    return 50, time.time() - t0


def main():
    print('===== T4 能耗平移：96M 混合架构（NVML 口径）=====')
    idle = SAMPLER.read_once()
    print(f'空闲功率: {idle:.1f} W')
    results = {'idle_w': round(idle, 1),
               'note': '4080 台式独显口径，绝对值不外推手机 SoC；结论看比值与量级'}

    arms = [
        ('infer_b1_hyb96_fp32', lambda: timed_infer(load_hyb(False), KW)),
        ('infer_b1_hyb96_fp16', lambda: timed_infer(load_hyb(True), KW)),
        ('infer_b1_tf101_fp32', lambda: timed_infer(load_tf(), None)),
        ('infer_b1_pcn21_fp32', lambda: timed_infer(load_pcn21(), KW21)),
        ('gen40_hyb96_fp32', lambda: timed_gen(load_hyb(False), KW)),
        ('gen40_hyb96_fp16', lambda: timed_gen(load_hyb(True), KW)),
    ]
    for tag, wl in arms:
        row = measure(SAMPLER, idle, wl, tag)
        results[tag] = row
        cooldown(SAMPLER, idle)
        torch.cuda.empty_cache()

    row = measure(SAMPLER, idle, lambda: adapt50_timed(load_hyb(False)),
                  'adapt50_hyb96_fp32')
    row['j_per_step_abs'] = row['j_per_tok_abs']  # tokens 位=步数
    row['total_j'] = round(row['mean_w'] * row['duration_s'], 0)
    row.pop('j_per_tok_abs'); row.pop('j_per_tok_net'); row.pop('tok_per_s')
    results['adapt50_hyb96_fp32'] = row

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 't4_hyb.json', 'w'), indent=2, ensure_ascii=False)
    print(f'输出: {OUT / "t4_hyb.json"}')


if __name__ == '__main__':
    main()
