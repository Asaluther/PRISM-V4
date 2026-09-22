#!/usr/bin/env python3
"""a3_twostage — int4 底座 + fp 适应头的两段式部署形态

问题（v7_demo 发现）：全 int4 fake-quant 下直接适应损失 ~10pp 增益
（+59.8% vs fp32 的 +70.1%）——量化后的 PCN 头权重适应能力受损。

两段式解法：TF 骨干（layers.*，适应中冻结）int4 量化；PCN 头
（pcn_blocks.*，适应的可训练部分）保持 fp32。预期恢复到 fp32 的 90%+。

三臂对照（同一检查点、同一用户数据、同一 50 步协议）：
  A. 全 fp32                    （基线，v7_demo 数字 +70.1%）
  B. 全 int4 fake-quant          （对照，v7_demo 数字 +59.8%）
  C. int4 骨干 + fp 头（两段式）   （本实验的主体）

预注册判据：C 的适应增益 ≥ A 的 90%（即 ≥ +63%）。

用法：.venv/Scripts/python.exe a3_twostage.py   (纯 CPU)
输出：results/v7/a3_twostage.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from bench_int4 import fake_quant_tensor
from src.models.hybrid import HybridModel
from src.data.tinystories import TinyStoriesDataset
import numpy as np

VOCAB = 50257
CKPT = 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'
KW = dict(topk=128, no_gating=True)
OUT = Path('results/v7')
GROUP = 64


def load():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


def quant_backbone_only(model):
    """只对 TF 骨干（layers.*）做 int4-g64 fake-quant；PCN 头保持 fp32"""
    quantized = 0
    for name, p in model.named_parameters():
        if not name.startswith('layers.') or 'weight' not in name or p.dim() != 2:
            continue
        if 'W_res' in name or 'emb' in name:
            continue
        if p.shape[1] % GROUP != 0:
            continue
        p.data = fake_quant_tensor(p.data, 4, GROUP, True)
        quantized += 1
    return model, quantized


def quant_all(model):
    """全模型 int4-g64 fake-quant（W_res + emb 豁免）——v7_demo --quant int4 的复现"""
    from bench_int4 import quant_model
    return quant_model(model, bits=4, group=GROUP, asym=True, exclude='W_res')


@torch.no_grad()
def eval_ppl(model, data):
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        logits = model(data['x'][i:i+1], **KW)
        l = F.cross_entropy(logits.reshape(-1, VOCAB),
                            data['y'][i:i+1].reshape(-1), reduction='sum')
        tot += l.item(); n += data['y'][i:i+1].numel()
    return math.exp(min(tot / n, 20))


def adapt50(model, user_data):
    """v7_demo 同协议：冻结 TF 骨干+emb，PCN 头 50 步 @5e-4"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = user_data['x'], user_data['y']
    model.train()
    for step in range(50):
        ci = step % xs.size(0)
        logits = model(xs[ci:ci+1], **KW)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                               xs[ci:ci+1].reshape(-1)[1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def make_data():
    ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    rows = np.stack([np.asarray(r) for r in ds.data[:8]])
    d = torch.from_numpy(rows.astype('int64'))
    user = {'x': d[:6, :-1].contiguous(), 'y': d[:6, 1:].contiguous()}
    test = {'x': d[6:8, :-1].contiguous(), 'y': d[6:8, 1:].contiguous()}
    return user, test


def main():
    print('===== A3：int4 底座 + fp 适应头（两段式 vs 全 fp32 vs 全 int4）=====')
    user, test = make_data()
    results = {}

    # A. 全 fp32
    m = load()
    p0 = eval_ppl(m, test)
    m = adapt50(m, user)
    p1 = eval_ppl(m, test)
    imp_fp32 = (1 - p1 / p0) * 100
    results['A_full_fp32'] = {'before': round(p0, 1), 'after': round(p1, 1),
                              'improvement_pct': round(imp_fp32, 1)}
    print(f'A 全 fp32:  {p0:.1f} → {p1:.1f} ({imp_fp32:+.1f}%)')
    del m

    # B. 全 int4 fake-quant
    m = load()
    m = quant_all(m)
    p0 = eval_ppl(m, test)
    m = adapt50(m, user)
    p1 = eval_ppl(m, test)
    imp_int4 = (1 - p1 / p0) * 100
    results['B_full_int4'] = {'before': round(p0, 1), 'after': round(p1, 1),
                              'improvement_pct': round(imp_int4, 1)}
    print(f'B 全 int4:  {p0:.1f} → {p1:.1f} ({imp_int4:+.1f}%)')
    del m

    # C. int4 骨干 + fp 头（两段式）
    m = load()
    m, n_quant = quant_backbone_only(m)
    p0 = eval_ppl(m, test)
    m = adapt50(m, user)
    p1 = eval_ppl(m, test)
    imp_2stage = (1 - p1 / p0) * 100
    results['C_int4_backbone_fp_head'] = {
        'n_quantized_layers': n_quant,
        'before': round(p0, 1), 'after': round(p1, 1),
        'improvement_pct': round(imp_2stage, 1)}
    print(f'C 两段式:   {p0:.1f} → {p1:.1f} ({imp_2stage:+.1f}%) '
          f'[量化 {n_quant} 层骨干]')
    del m

    # 判据
    recovery = imp_2stage / imp_fp32 * 100 if imp_fp32 > 0 else 0
    results['criteria'] = {
        '两段式恢复 ≥90% fp32': bool(recovery >= 90),
        'recovery_pct': round(recovery, 1),
        'fp32_baseline': round(imp_fp32, 1),
        'int4_full': round(imp_int4, 1),
        'twostage': round(imp_2stage, 1)}
    print(f'\n恢复率: {recovery:.1f}% (判据 ≥90%: '
          f'{"✅" if recovery >= 90 else "❌"})')

    # 体积估算
    total = sum(p.numel() for p in load().parameters())
    backbone = sum(p.numel() for n, p in load().named_parameters()
                   if n.startswith('layers.'))
    head = total - backbone
    emb = sum(p.numel() for n, p in load().named_parameters() if 'emb' in n)
    mb_2stage = (backbone * 0.5 + head * 4 + emb * 4) / 1024 / 1024
    results['size_est_mb'] = {
        'fp32': round(total * 4 / 1024 / 1024, 1),
        'full_int4': round((total - emb) * 0.5 / 1024 / 1024 + emb * 4 / 1024 / 1024, 1),
        'twostage': round(mb_2stage, 1)}
    print(f'体积: fp32 {results["size_est_mb"]["fp32"]} MB | '
          f'全 int4 {results["size_est_mb"]["full_int4"]} MB | '
          f'两段式 {results["size_est_mb"]["twostage"]} MB')

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'a3_twostage.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'输出: {OUT / "a3_twostage.json"}')


if __name__ == '__main__':
    main()
