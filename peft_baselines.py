#!/usr/bin/env python3
"""peft_baselines — A1：LoRA / bitfit 少样本适应基线（响应 V5 审稿 W2）

审稿人质疑：TF 在适应协议中的灾难性过拟合可能只是「没上 PEFT」。本脚本在
与既有协议完全相同的设定下补齐 PEFT 基线：

对照架构与检查点（与 seed_replication 相同，seed 0）：
  tf  = wt_200m_tf_lr5e-4_ckpt    hyb = wt_094m_hyb_s0

适应范围（与 full-FT 协议的可训练范围一致）：
  tf : 顶部 12/24 TF 块（底部 12 + emb 冻结）
  hyb: 全部 PCN 头（TF 骨干 + emb 冻结——阶段②语义）

方法：
  lora  : 该范围内所有 nn.Linear 递归包 LoRA（r=8/α=16 默认；TF 另扫 r∈{4,16}），
          基座全冻只训 A/B；B 零初始化（注入即恒等，起点行为与基座严格一致）
  bitfit: 该范围内只训 bias + LayerNorm 仿射参数

预注册判据（跑前立好）：
  ① 若 PEFT-TF 单发改善 ≥ +30%（达 HYB full-FT 档）→ 适应优势主张需重述为
    「免调参自限适应」而非「架构独有」；若 < +30% → 架构优势经受住 PEFT 对照
  ② 流式同问（对照 HYB full-FT@5e-5 的 +43.0%）
  ③ 附带报告 ΔW 代理（适配器 B@A 的 L2）与墙钟时间

用法：.venv/Scripts/python.exe peft_baselines.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from coldstart_090m import PCN_KW, VOCAB, DEV, make_data
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel

OUT = Path('results/v7')
CKPTS = {'tf': 'results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
         'hyb': 'results/wt_094m_hyb_s0/best_model.pt'}


def load_base(kind):
    if kind == 'hyb':
        m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                        n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                        d_gate=128, dropout=0.0, max_seq_len=256)
    else:
        m = TransformerModel(vocab_size=VOCAB, d_model=512, n_layers=24,
                             n_heads=4, ffn_dim=2048, dropout=0.0,
                             max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load(CKPTS[kind], map_location=DEV, weights_only=True))
    return m.to(DEV)


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r)·xAᵀBᵀ；B 零初始化 → 注入即恒等"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)


def scope_blocks(model, kind):
    """full-FT 协议的可训练范围：tf=顶部 12 块 / hyb=全部 PCN 头"""
    if kind == 'hyb':
        return list(model.pcn_blocks)
    return [blk for i, blk in enumerate(model.layers) if i >= 12]


def setup_peft(model, kind, method, rank=8):
    """注入/解冻并返回 (trainable, 注入数)；完成后模型行为与基座一致"""
    blocks = scope_blocks(model, kind)
    if method == 'lora':
        # 先冻结全部，再注入——否则 A/B 也会被冻结（冒烟测试抓到的顺序 bug）
        for p in model.parameters():
            p.requires_grad = False
        def wrap(parent):
            n = 0
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.MultiheadAttention):
                    continue   # MHA 的 out_proj 不能包（内部直取 .weight）
                if isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, r=rank))
                    n += 1
                else:
                    n += wrap(child)
            return n
        wrapped = sum(wrap(b) for b in blocks)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return trainable, wrapped
    if method == 'bitfit':
        for p in model.parameters():
            p.requires_grad = False
        n_free = 0
        for b in blocks + [model.ln_out]:
            for nm, p in b.named_parameters():
                if nm.endswith('bias') or nm.endswith('.weight') and \
                        isinstance(dict(b.named_modules()).get(nm.rsplit('.', 1)[0]),
                                   nn.LayerNorm):
                    p.requires_grad = True
                    n_free += 1
        trainable = [p for p in model.parameters() if p.requires_grad]
        return trainable, n_free
    raise ValueError(method)


def forward(model, kind, x):
    return model(x, **PCN_KW) if kind == 'hyb' else model(x)


@torch.no_grad()
def eval_ppl(model, data, kind):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = forward(model, kind, x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def adapt(model, kind, data, trainable, steps, lr):
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    xs, ys = data['x'].to(DEV), data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = forward(model, kind, x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def adapter_dw(model):
    dws = [(m.B @ m.A).norm().item() for m in model.modules()
           if isinstance(m, LoRALinear)]
    return round(sum(dws) / len(dws), 4) if dws else 0.0


def coldstart_run(kind, method, rank, lr, users, tests):
    imps, walls, dws, befores, afters = [], [], [], [], []
    for u in range(len(users)):
        m = load_base(kind)
        trainable, _ = setup_peft(m, kind, method, rank or 8)
        pb = eval_ppl(m, tests[u], kind)
        befores.append(round(pb, 1))
        t0 = time.time()
        m = adapt(m, kind, users[u], trainable, 300, lr)
        walls.append(time.time() - t0)
        pa = eval_ppl(m, tests[u], kind)
        afters.append(round(pa, 1))
        imps.append((1 - pa / pb) * 100)
        dws.append(adapter_dw(m))
        del m; torch.cuda.empty_cache()
    return {'before': befores, 'after': afters,
            'per_user': [round(x, 1) for x in imps],
            'avg': round(statistics.mean(imps), 1),
            'dw_proxy': round(statistics.mean(dws), 4),
            'wall_s': round(statistics.mean(walls), 1)}


def main():
    print('===== A1：PEFT 基线（LoRA / bitfit）=====\n')
    users, tests = make_data()
    results = {}

    # --- 单发：kind × method × lr ---
    grid = [('tf', 'lora', 8, 5e-4), ('tf', 'lora', 8, 1e-3),
            ('tf', 'bitfit', None, 5e-4), ('tf', 'bitfit', None, 1e-3),
            ('hyb', 'lora', 8, 5e-4), ('hyb', 'lora', 8, 1e-3),
            ('hyb', 'bitfit', None, 5e-4), ('hyb', 'bitfit', None, 1e-3)]
    best = {}
    for kind, method, rank, lr in grid:
        r = coldstart_run(kind, method, rank, lr, users, tests)
        key = f'{kind}/{method}' + (f'/r{rank}' if rank else '') + f'@lr{lr:g}'
        results[key] = r
        print(f'  {key:26s} 平均 {r["avg"]:+7.1f}%  ΔW代理 {r["dw_proxy"]}  '
              f'{r["wall_s"]:.0f}s')
        k = (kind, method)
        if k not in best or r['avg'] > best[k][0]:
            best[k] = (r['avg'], lr, rank)

    json.dump(results, open(OUT / 'peft_baselines.json', 'w'),
              indent=2, ensure_ascii=False)   # 增量保存：冷启动网格完成即落盘

    # --- TF LoRA rank 扫描（最优 lr 上）---
    tf_lr = best[('tf', 'lora')][1]
    for rank in (4, 16):
        r = coldstart_run('tf', 'lora', rank, tf_lr, users, tests)
        results[f'tf/lora/r{rank}@lr{tf_lr:g}'] = r
        print(f'  tf/lora/r{rank}@lr{tf_lr:g}         平均 {r["avg"]:+7.1f}%')
        if r['avg'] > best[('tf', 'lora')][0]:
            best[('tf', 'lora')] = (r['avg'], tf_lr, rank)

    # --- 流式：每 kind 用 LoRA 最优配置 @ {5e-5, 1e-4} ---
    print('\n--- 流式（LoRA 最优配置）---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    tok = GPT2TokenizerFast.from_pretrained(str(Path(__file__).parent / 'tokenizer'))
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    for kind in ('tf', 'hyb'):
        _, lr_cs, rank = best[(kind, 'lora')]
        m0 = load_base(kind)
        frozen = [eval_ppl(m0, b, kind) for b in stream]
        held0 = eval_ppl(m0, user_held, kind)
        del m0; torch.cuda.empty_cache()
        for lr_st in (5e-5, 1e-4):
            m = load_base(kind)
            trainable, _ = setup_peft(m, kind, 'lora', rank or 8)
            traj = []
            for i, batch in enumerate(stream):
                traj.append(eval_ppl(m, batch, kind))
                if i < len(stream) - 1:
                    m = adapt(m, kind, batch, trainable, 60, lr_st)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held = eval_ppl(m, user_held, kind)
            held_gain = (1 - held / held0) * 100
            key = f'stream/{kind}/lora@lr{lr_st:g}'
            results[key] = {'online_gain_pct': round(gain, 1),
                            'heldout_gain_pct': round(held_gain, 1)}
            print(f'  {key:32s} 在线 {gain:+.1f}%  held-out {held_gain:+.1f}%')
            del m; torch.cuda.empty_cache()

    json.dump(results, open(OUT / 'peft_baselines.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "peft_baselines.json"}')

    print('\n===== 预注册判据 =====')
    tf_best_cs = max(v['avg'] for k, v in results.items()
                     if k.startswith('tf/') and not k.startswith('stream'))
    tf_best_st = max(v['online_gain_pct'] for k, v in results.items()
                     if k.startswith('stream/tf/'))
    v1 = '≥ +30% → 适应优势主张需重述为「免调参自限」' if tf_best_cs >= 30 \
        else '< +30% → 架构优势经受住 PEFT 对照'
    print(f'  ① PEFT-TF 单发最佳 {tf_best_cs:+.1f}%：{v1}')
    print(f'  ② PEFT-TF 流式最佳 {tf_best_st:+.1f}%（对照 HYB full-FT@5e-5 +43.0%）')


if __name__ == '__main__':
    main()
