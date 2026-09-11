#!/usr/bin/env python3
"""T6 — int4 权重量化敏感性（GOAL V7#2 端侧压缩前置数据）

T2 结论回顾：int8 per-channel 对称 RTN 无损（TF +0.0% / NG +0.1%）。
本基准问：压到 4bit 还剩多少？PCN 误差流权重比 TF 更耐还是更敏感？

方法（假量化敏感性，精度口径；不含 int4 kernel 收益）：
  阶梯：int8 对称（T2 连续性格）/ int4 对称 per-channel / int4 非对称 per-channel
        / int4 非对称 group128 / int4 非对称 group64
  RTN（round-to-nearest），与 T2 同族；若显著损失，GPTQ 误差补偿另行立项
公平性（教训 8）：WT 对用公平 checkpoint（wt_tf_lr1e3_s1，PPL 215；
  T2 的 wt_causal_transformer_s0 为 lr 次优 PPL 557，不再用作结论依据）。
  另加 TS 域内对（终端真实场景：部署已落在用户域的模型）。

判读口径：<1% 近无损 / <5% 可用 / >20% 需 GPTQ 级方法。
注意：n 计数修正为全 y（T2 的 ye[:,1:] 是历史口径，绝对 PPL 有 ~0.4% 偏置，
  loss_pct 不受影响；本脚本 int8 格与 T2 结论交叉验证）。

用法：.venv/Scripts/python.exe bench_int4.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.wikitext import WikiTextDataset
from src.data.tinystories import TinyStoriesDataset

DEV = torch.device('cuda')
OUT = Path('results/efficiency')


def fake_quant_tensor(W, bits, group, asym):
    """RTN 假量化一个 [out, in] 权重，返回量化-反量化后的 fp 权重"""
    out_dim, in_dim = W.shape
    assert in_dim % group == 0, f'in_dim {in_dim} 不被 group {group} 整除'
    Wg = W.reshape(out_dim, in_dim // group, group)
    if asym:
        mx = Wg.max(dim=-1, keepdim=True).values
        mn = Wg.min(dim=-1, keepdim=True).values
        scale = (mx - mn) / (2 ** bits - 1)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zp = torch.round(-mn / scale)
        q = torch.clamp(torch.round(Wg / scale) + zp, 0, 2 ** bits - 1)
        Wq = (q - zp) * scale
    else:
        mx = Wg.abs().max(dim=-1, keepdim=True).values
        scale = mx / (2 ** (bits - 1) - 1)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(Wg / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
        Wq = q * scale
    return Wq.reshape(out_dim, in_dim)


def quant_model(model, bits, group, asym, exclude=None):
    """对所有 2D 权重（embedding 除外，与 T2 口径一致）做 RTN 假量化。
    group=None 表示 per-channel（每行一组，group=in_dim）。
    exclude: 'W_res' = W_res 除外（保 fp）；'only_W_res' = 仅量化 W_res（诊断用）。"""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'weight' not in name or p.dim() != 2 or 'emb' in name:
                continue
            if exclude == 'W_res' and 'W_res' in name:
                continue
            if exclude == 'only_W_res' and 'W_res' not in name:
                continue
            g = p.shape[1] if group is None else group
            p.copy_(fake_quant_tensor(p.data.float(), bits, g, asym))
    return m


def size_mb(model, bits, group, asym, exclude=None):
    """理论体积：量化权重 @bits/8 + 每组 scale/zero(fp32)，embedding/norms 保持 fp16。
    group=None 表示 per-channel（每行一组）。exclude 语义同 quant_model（除外者按 fp16 计）。"""
    q_bytes, o_bytes = 0, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            n = p.numel()
            is_linear = 'weight' in name and p.dim() == 2 and 'emb' not in name
            if exclude == 'W_res' and 'W_res' in name:
                is_linear = False
            if exclude == 'only_W_res' and 'W_res' not in name:
                is_linear = False
            if is_linear:
                g = p.shape[1] if group is None else group
                groups = (p.shape[1] // g) * p.shape[0]
                scale_bytes = groups * 4 * (2 if asym else 1)
                q_bytes += n * bits // 8 + scale_bytes
            else:
                o_bytes += n * 2  # fp16
    return round((q_bytes + o_bytes) / 1e6, 1)


@torch.no_grad()
def val_ppl(model, data, kind):
    tot, n = 0.0, 0
    for x, y in data:
        xe, ye = x.to(DEV), y.to(DEV)
        if kind == 'tf':
            logits = model(xe)
        else:
            logits = model(xe, topk=64, no_gating=True)
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            ye.reshape(-1), reduction='sum')
        tot += l.item(); n += ye.numel()
    return math.exp(min(tot / n, 20))


METHODS = [
    ('int8 对称 per-channel (T2 连续)', 8, None, False),
    ('int4 对称 per-channel', 4, None, False),
    ('int4 非对称 per-channel', 4, None, True),
    ('int4 非对称 group128', 4, 128, True),
    ('int4 非对称 group64', 4, 64, True),
]


def build_pair(domain):
    if domain == 'wt':
        tf = TransformerModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                              ffn_dim=1024, dropout=0.0, max_seq_len=256, attn_impl='sdpa')
        tf_ckpt = 'results/wt_tf_lr1e3_s1/best_model.pt'
        ng = PCNModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                      d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed')
        ng_ckpt = 'results/wt_causal_no_gating_s1/best_model.pt'
        ds = WikiTextDataset(split='validation', seq_len=256)
        val = [(ds[i][0].unsqueeze(0), ds[i][1].unsqueeze(0)) for i in range(50)]
    else:
        tf = TransformerModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                              ffn_dim=1024, dropout=0.0, max_seq_len=256, attn_impl='sdpa')
        tf_ckpt = 'results/ts_tf_lr1e3_s1/best_model.pt'
        ng = PCNModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                      d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed')
        ng_ckpt = 'results/ts_causal_no_gating_s1/best_model.pt'
        ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
        val = [(ds[i][0].unsqueeze(0), ds[i][1].unsqueeze(0)) for i in range(50)]
    arms = []
    for tag, m, kind, ckpt in (('TF', tf, 'tf', tf_ckpt), ('PCN no_gating', ng, 'ng', ng_ckpt)):
        sd = torch.load(ckpt, map_location=DEV, weights_only=True)
        m.load_state_dict(sd)
        arms.append((tag, m.to(DEV).eval(), kind))
    return arms, val


def main():
    print('===== T6: int4 weight-only 量化敏感性（公平 checkpoint 对）=====\n')
    results = {}
    for domain in ('wt', 'ts'):
        arms, val = build_pair(domain)
        results[domain] = {}
        print(f'--- {domain.upper()} 域 ---')
        for tag, model, kind in arms:
            p_fp = val_ppl(model, val, kind)
            row = {'fp32': round(p_fp, 2), 'methods': {}, 'size_mb': {}}
            fp_mb = size_mb(model, 32, None, False)
            row['size_mb']['fp32(全fp16对照)'] = fp_mb
            print(f'  [{tag}] fp32 PPL = {p_fp:.2f}')
            for mname, bits, group, asym in METHODS:
                mq = quant_model(model, bits, group, asym)
                p_q = val_ppl(mq, val, kind)
                loss = (p_q / p_fp - 1) * 100
                row['methods'][mname] = {'ppl': round(p_q, 2), 'loss_pct': round(loss, 1)}
                row['size_mb'][mname] = size_mb(model, bits, group, asym)
                print(f'    {mname:<28} PPL {p_q:8.2f}  ({loss:+6.1f}%)  '
                      f'~{row["size_mb"][mname]} MB')
            results[domain][tag] = row
        print()

    json.dump(results, open(OUT / 't6_int4.json', 'w'), indent=2, ensure_ascii=False)

    # 假设验证 + 部署配方探针：W_res 主干假设
    # 假设：PCN 残差主干穿过 W_res matmul（非加性恒等），量化误差直接进信号路径跨层复合；
    #       TF 的加性残差把误差限制在增量支路 → int4 无损。
    # 探针：a) 仅量化 W_res（若单独造成大损失 → 主干假设成立）
    #       b) W_res 除外全量化（若恢复近无损 → 部署配方 = int4-g64 + W_res 保 fp16）
    print('\n===== W_res 主干假设探针（PCN only）=====')
    probe = {}
    for domain in ('wt', 'ts'):
        arms, val = build_pair(domain)
        model = next(m for t, m, k in arms if t == 'PCN no_gating')
        kind = 'ng'
        p_fp = results[domain]['PCN no_gating']['fp32']
        probe[domain] = {'fp32': p_fp}
        for pname, exclude in (('仅 W_res int4-g64', 'only_W_res'),
                               ('int4-g64 W_res除外', 'W_res')):
            mq = quant_model(model, 4, 64, True, exclude=exclude)
            p_q = val_ppl(mq, val, kind)
            loss = (p_q / p_fp - 1) * 100
            sm = size_mb(model, 4, 64, True, exclude=exclude)
            probe[domain][pname] = {'ppl': round(p_q, 2), 'loss_pct': round(loss, 1),
                                    'size_mb': sm}
            print(f'  {domain.upper()} {pname:<20} PPL {p_q:8.2f}  ({loss:+6.1f}%)  ~{sm} MB')

    results['wres_probe'] = probe
    json.dump(results, open(OUT / 't6_int4.json', 'w'), indent=2, ensure_ascii=False)

    # 判读
    print('===== 判读（<1% 近无损 / <5% 可用 / >20% 需 GPTQ）=====')
    best = 'int4 非对称 group128'
    for domain in ('wt', 'ts'):
        for tag in ('TF', 'PCN no_gating'):
            l = results[domain][tag]['methods'][best]['loss_pct']
            l8 = results[domain][tag]['methods']['int8 对称 per-channel (T2 连续)']['loss_pct']
            print(f'  {domain.upper()} {tag:<14} int8 {l8:+.1f}%  int4-g128 {l:+.1f}%')
    print(f'\n输出: {OUT}/t6_int4.json')


if __name__ == '__main__':
    main()
