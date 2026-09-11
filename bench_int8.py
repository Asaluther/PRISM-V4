#!/usr/bin/env python3
"""T2 — int8 权重量化敏感性（终端部署必答题）"""
import sys, json, math, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.wikitext import WikiTextDataset

DEV = torch.device('cuda')


def quant_int8(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'weight' in name and p.dim() == 2 and 'emb' not in name:
                for i in range(p.shape[0]):
                    row = p[i]
                    scale = row.abs().max() / 127.0
                    if scale > 0:
                        q = torch.clamp(torch.round(row / scale), -127, 127)
                        p[i] = q * scale
    return m


@torch.no_grad()
def val_ppl(model, data, kind):
    tot, n = 0.0, 0
    for x, y in data:
        xe, ye = x.to(DEV), y.to(DEV)
        if kind == 'tf':
            logits = model(xe)
        else:
            kw = {'no_gating': True} if kind == 'ng' else {}
            logits = model(xe, topk=64, **kw)
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            ye.reshape(-1), reduction='sum')
        tot += l.item(); n += ye[:, 1:].numel()
    return math.exp(min(tot / n, 20))


def load_and_eval(tag, build, kind, name, val):
    fp = build()
    sd = torch.load(f'results/{name}/best_model.pt', map_location=DEV, weights_only=True)
    fp.load_state_dict(sd); fp = fp.to(DEV).eval()
    p_fp = val_ppl(fp, val, kind)
    q = quant_int8(fp).to(DEV).eval()
    p_q = val_ppl(q, val, kind)
    loss = (p_q / p_fp - 1) * 100
    print(f'  {tag:<16} fp32={p_fp:8.2f}  int8={p_q:8.2f}  损失={loss:+.1f}%')
    return {'fp32': round(p_fp, 2), 'int8': round(p_q, 2), 'loss_pct': round(loss, 1)}


def main():
    Path('results/efficiency').mkdir(exist_ok=True)
    ds = WikiTextDataset(split='validation', seq_len=256)
    val = [(ds[i][0].unsqueeze(0), ds[i][1].unsqueeze(0)) for i in range(50)]
    print('===== T2: int8 weight-only 量化敏感性（WT 因果 checkpoint）=====')
    res = {}
    res['transformer'] = load_and_eval(
        'Transformer', lambda: TransformerModel(vocab_size=50257, d_model=256, n_layers=12,
                                                n_heads=4, ffn_dim=1024, dropout=0.0,
                                                max_seq_len=256, attn_impl='mha'),
        'tf', 'wt_causal_transformer_s0', val)
    res['no_gating'] = load_and_eval(
        'PCN no_gating', lambda: PCNModel(vocab_size=50257, d_model=256, n_layers=12,
                                          n_heads=4, d_gate=64, dropout=0.0,
                                          max_seq_len=256, init_mode='fixed'),
        'ng', 'wt_causal_no_gating_s0', val)
    json.dump(res, open('results/efficiency/t2_int8.json', 'w'), indent=2)
    d_tf, d_ng = res['transformer']['loss_pct'], res['no_gating']['loss_pct']
    verdict = 'PCN 更耐量化（终端优势）' if d_ng < d_tf else 'PCN 更敏感（工程边界）'
    print(f'\n判读: TF {d_tf:+.1f}% vs NG {d_ng:+.1f}% -> {verdict}')


if __name__ == '__main__':
    main()
