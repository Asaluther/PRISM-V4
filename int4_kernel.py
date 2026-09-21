#!/usr/bin/env python3
"""int4_kernel — packed int4 真量化实测（路线 A：torchao；失败降级 B：自写打包）

fake-quant 时代（int4_gate.py）只验证了数值与解析体积；本脚本把权重真正
打包成 int4 常驻内存，实测：权重驻留内存（tensor 存储实测）、进程 RSS 三段
对比、真实生成/推理速度（允许反量化开销）、PPL 损失复测。

预注册判据：权重驻留 ≤110MB、RSS 较 fp32 降 ≥30%、生成 ≥25 tok/s、
PPL 损失 ≤1%。任一不过则如实报告（判据按路线 A 的布局预设；路线 B 下
embedding 保持 fp32，权重驻留口径单独汇报）。

用法：.venv/Scripts/python.exe int4_kernel.py   (纯 CPU)
输出：results/v7/int4_kernel.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

PROC = psutil.Process()
VOCAB = 50257
CKPT = 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'
KW = dict(topk=128, no_gating=True)
OUT = Path('results/v7')


def rss_mb():
    gc.collect()
    return PROC.memory_info().rss / 1024 / 1024


def weight_storage_mb(model):
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        try:
            total += p.untyped_storage().nbytes()
        except Exception:
            total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.untyped_storage().nbytes()
    return total / 1024 / 1024


@torch.no_grad()
def eval_ppl(model, data):
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        logits = model(data['x'][i:i+1], **KW)
        l = F.cross_entropy(logits.reshape(-1, VOCAB),
                            data['y'][i:i+1].reshape(-1), reduction='sum')
        tot += l.item(); n += data['y'][i:i+1].numel()
    return math.exp(min(tot / n, 20))


@torch.no_grad()
def bench_infer(model, steps=10):
    x = torch.randint(0, VOCAB, (1, 256))
    for _ in range(3):
        model(x, **KW)
    t0 = time.time()
    for _ in range(steps):
        model(x, **KW)
    return steps * 256 / (time.time() - t0)


@torch.no_grad()
def bench_gen(model, max_new=40):
    ids = torch.randint(0, VOCAB, (1, 20))
    model(ids[:, -256:], **KW)
    t0 = time.time()
    for _ in range(max_new):
        logits = model(ids[:, -256:], **KW)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    dt = time.time() - t0
    return max_new / dt, dt / max_new * 1000


class PackedInt4Linear(nn.Module):
    """路线 B：int4-g64 非对称打包（两权重/字节 uint8）+ 前向反量化。
    权重常驻内存真实下降；计算开销为反量化（速度近似 fp32）。"""

    def __init__(self, linear: nn.Linear, group: int = 64):
        super().__init__()
        W = linear.weight.detach()
        out, inp = W.shape
        assert inp % group == 0
        g = W.reshape(out, inp // group, group)
        mn = g.amin(-1, keepdim=True)
        mx = g.amax(-1, keepdim=True)
        scale = ((mx - mn) / 15).clamp(min=1e-12)
        zp = torch.round(-mn / scale).clamp(0, 15)
        q = torch.round(g / scale + zp).clamp(0, 15).to(torch.uint8)
        q = q.reshape(out, inp)
        self.register_buffer('packed',
                             (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous())
        self.register_buffer('scale',
                             scale.half().reshape(out, inp // group).contiguous())
        self.register_buffer('zp',
                             zp.reshape(out, inp // group).to(torch.uint8).contiguous())
        self.in_features, self.out_features = inp, out
        self.bias = linear.bias

    def forward(self, x):
        q = torch.empty(self.packed.shape[0], self.packed.shape[1] * 2,
                        dtype=torch.uint8, device=x.device)
        q[:, 0::2] = self.packed & 0xF
        q[:, 1::2] = self.packed >> 4
        W = (q.reshape(self.out_features, -1, 64).float()
             - self.zp.float().unsqueeze(-1)) * self.scale.float().unsqueeze(-1)
        return F.linear(x, W.reshape(self.out_features, -1), self.bias)


def pack_model(model):
    _exempt = ('W_res', 'w_g')   # w_g 前向直取 .weight（pcn.py:117），豁免

    def _pack(mod):
        n = 0
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                continue   # MHA 内部直取 out_proj.weight，视为原子不打包
            if isinstance(child, nn.Linear) and not any(e in name for e in _exempt):
                setattr(mod, name, PackedInt4Linear(child))
                n += 1
            else:
                n += _pack(child)
        return n
    return _pack(model)


def main():
    print('===== packed int4 真量化实测（纯 CPU）=====')
    from src.models.hybrid import HybridModel
    from src.data.wikitext import WikiTextDataset

    wt = WikiTextDataset(split='validation', seq_len=256)
    data = {'x': torch.from_numpy(wt.data[:100, :-1].astype('int64')),
            'y': torch.from_numpy(wt.data[:100, 1:].astype('int64'))}
    results = {}

    rss0 = rss_mb()
    print(f'RSS 基线（无模型）: {rss0:.0f} MB')

    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    m = m.eval()
    rss_fp32 = rss_mb()
    w_fp32 = weight_storage_mb(m)
    ppl_fp32 = eval_ppl(m, data)
    tps_fp32 = bench_infer(m)
    gen_fp32, ms_fp32 = bench_gen(m)
    print(f'fp32: RSS {rss_fp32:.0f} MB | 权重 {w_fp32:.0f} MB | PPL {ppl_fp32:.2f} '
          f'| 推理 {tps_fp32:.0f} tok/s | 生成 {gen_fp32:.1f} tok/s')
    results['fp32'] = {'rss_mb': round(rss_fp32), 'weights_mb': round(w_fp32),
                       'ppl': round(ppl_fp32, 2), 'infer_tok_s': round(tps_fp32),
                       'gen_tok_s': round(gen_fp32, 1),
                       'gen_ms_per_tok': round(ms_fp32, 1)}

    # ---- 量化：路线 A（torchao）→ ImportError 降级 B（自写打包） ----
    try:
        from torchao.quantization import quantize_, Int4WeightOnlyConfig
        quantize_(m, Int4WeightOnlyConfig(),
                  filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear)
                  and 'W_res' not in fqn)
        route = 'A: torchao Int4WeightOnly'
    except ImportError:
        n_packed = pack_model(m)
        route = f'B: 自写打包 int4-g64（{n_packed} Linear，mslk 不可用降级）'
    print(f'量化路线: {route}')

    rss_q = rss_mb()
    w_q = weight_storage_mb(m)
    ppl_q = eval_ppl(m, data)
    tps_q = bench_infer(m)
    gen_q, ms_q = bench_gen(m)
    loss = (ppl_q - ppl_fp32) / ppl_fp32 * 100
    rss_drop = (rss_fp32 - rss_q) / rss_fp32 * 100
    print(f'int4: RSS {rss_q:.0f} MB（降 {rss_drop:.0f}%）| 权重驻留 {w_q:.0f} MB '
          f'| PPL {ppl_q:.2f}（{loss:+.2f}%）| 推理 {tps_q:.0f} tok/s | '
          f'生成 {gen_q:.1f} tok/s')
    results['int4_packed'] = {
        'route': route,
        'rss_mb': round(rss_q), 'weights_mb': round(w_q),
        'ppl': round(ppl_q, 2), 'loss_pct': round(loss, 2),
        'infer_tok_s': round(tps_q), 'gen_tok_s': round(gen_q, 1),
        'gen_ms_per_tok': round(ms_q, 1),
        'rss_drop_pct': round(rss_drop, 1)}

    crit = {
        '权重驻留 ≤110MB': w_q <= 110,
        'RSS 较 fp32 降 ≥30%': rss_drop >= 30,
        '生成 ≥25 tok/s': gen_q >= 25,
        'PPL 损失 ≤1%': loss <= 1.0,
    }
    print('\n判据记分板:')
    for k, v in crit.items():
        print(f'  {k}: {"✅" if v else "❌"}')
    results['criteria'] = {k: bool(v) for k, v in crit.items()}
    results['rss_baseline_mb'] = round(rss0)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'int4_kernel.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'输出: {OUT / "int4_kernel.json"}')


if __name__ == '__main__':
    main()
