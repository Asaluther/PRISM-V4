#!/usr/bin/env python3
"""int4_gate — V7 demo 前置门槛实验：int4 配方扩展到 96M 混合架构

T6 配方（21M 小模型验证）：int4-g64 非对称 + W_res 豁免 + embedding 豁免 ≈ 近无损。
本实验在 h2e4 终端默认检查点（96M 混合，d512）上复测量化损失与体积。

预注册判据：PPL 损失 ≤1% → demo 支持 int4 模式；超限 → 如实报告，demo 保留
fp32/fp16 模式（判据失败不阻塞 demo，量化数字本身是新数据点）。

用法：.venv/Scripts/python.exe int4_gate.py   (纯 CPU)
输出：results/v7/int4_gate.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from bench_int4 import quant_model, size_mb
from src.models.hybrid import HybridModel
from src.data.wikitext import WikiTextDataset

VOCAB = 50257
CKPT = 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'
KW = dict(topk=128, no_gating=True)
OUT = Path('results/v7')


def load():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12, n_pcn_layers=12,
                    n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


@torch.no_grad()
def eval_ppl(model, data):
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        logits = model(data['x'][i:i+1], **KW)
        l = F.cross_entropy(logits.reshape(-1, VOCAB),
                            data['y'][i:i+1].reshape(-1), reduction='sum')
        tot += l.item(); n += data['y'][i:i+1].numel()
    return math.exp(min(tot / n, 20))


def main():
    print('===== int4 × 混合架构门槛实验（纯 CPU）=====')
    wt = WikiTextDataset(split='validation', seq_len=256)
    data = {'x': torch.from_numpy(wt.data[:100, :-1].astype('int64')),
            'y': torch.from_numpy(wt.data[:100, 1:].astype('int64'))}

    m = load()
    base = eval_ppl(m, data)
    print(f'基线 fp32 PPL: {base:.2f}')

    results = {'ckpt': CKPT, 'n_eval_seqs': 100, 'fp32_ppl': round(base, 2)}

    for tag, kw in [
        ('int4_g64_asym_Wres_exempt', dict(bits=4, group=64, asym=True, exclude='W_res')),
        ('int8_perchannel', dict(bits=8, group=None, asym=False, exclude=None)),
    ]:
        mq = quant_model(m, **kw)
        ppl = eval_ppl(mq, data)
        loss = (ppl - base) / base * 100
        mb_q = size_mb(m, **kw)
        results[tag] = {'ppl': round(ppl, 2), 'loss_pct': round(loss, 2),
                        'size_mb': round(mb_q, 1)}
        print(f'{tag:28s}: PPL {ppl:7.2f} ({loss:+.2f}%)  {mb_q:.0f} MB')
        del mq

    fp32_mb = sum(p.numel() for p in m.parameters()) * 4 / 1024 / 1024
    results['fp32_size_mb'] = round(fp32_mb, 1)
    gate = results['int4_g64_asym_Wres_exempt']['loss_pct'] <= 1.0
    results['gate_pass'] = bool(gate)
    print(f"\n判据（int4 损失 ≤1%）: {'✅ 通过' if gate else '❌ 超限——demo 保留 fp32/fp16 模式'}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'int4_gate.json', 'w'), indent=2, ensure_ascii=False)
    print(f'输出: {OUT / "int4_gate.json"}')


if __name__ == '__main__':
    main()
