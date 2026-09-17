#!/usr/bin/env python3
"""phase2_protocols — 阶段②变体的适应协议测量

对 4 个分割比/头lr 变体（16+8 / 8+16 / 头5e-5 / 头2e-4）跑单发 + 流式@{5e-5,1e-4}。

与 seed_replication 的关键差异——冻结语义：本脚本冻结**全部 TF 骨干**（model.layers
整表）而非固定前 12 层。原因：16+8 配置下沿用常数 12 会漏冻 4 层 TF，把「TF 层参与
适应」这个已知灾难源混进分割比对比。对 12+12 基线两种语义等价（其 layers 恰为 12），
已测数字可直接对照。

用法：.venv/Scripts/python.exe phase2_protocols.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

from coldstart_090m import PCN_KW, VOCAB, DEV, OUT, make_data
from streaming_090m import eval_ppl as st_eval, STEPS_PER_BATCH
from src.models.hybrid import HybridModel
from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
from src.data.tinystories import TinyStoriesDataset
from src.data.wikitext import WikiTextDataset

ST_LRS = [5e-5, 1e-4]

VARIANTS = {
    'hyb16_8':  dict(tf=16, pcn=8,  json='results/wt_097m_hyb16_8_s0/results.json',
                     ckpt='results/wt_097m_hyb16_8_s0/best_model.pt'),
    'hyb8_16':  dict(tf=8,  pcn=16, json='results/wt_094m_hyb8_16_s0/results.json',
                     ckpt='results/wt_094m_hyb8_16_s0/best_model.pt'),
    'hyb_h5e5': dict(tf=12, pcn=12, json='results/wt_096m_hyb_h5e-5_s0/results.json',
                     ckpt='results/wt_096m_hyb_h5e-5_s0/best_model.pt'),
    'hyb_h2e4': dict(tf=12, pcn=12, json='results/wt_096m_hyb_h2e-4_s0/results.json',
                     ckpt='results/wt_096m_hyb_h2e-4_s0/best_model.pt'),
}


def load_variant(name):
    v = VARIANTS[name]
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=v['tf'],
                    n_pcn_layers=v['pcn'], n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(v['ckpt'], map_location=DEV, weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def eval_ppl(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, **PCN_KW)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return __import__('math').exp(min(tot / n, 20))


def freeze_backbone(model):
    """冻结全部 TF 骨干 + embedding；适应只发生在 PCN 头（终端设计意图）"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def finetune(model, train_data, steps=300, lr=5e-4):
    trainable = freeze_backbone(model)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, **PCN_KW)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


def run_stream(model, stream, user_held, lr):
    pre = {n: p.detach().clone() for n, p in model.named_parameters()
           if n.startswith('pcn_blocks.')}
    traj = []
    for i, batch in enumerate(stream):
        traj.append(round(eval_ppl(model, batch), 2))
        if i < len(stream) - 1:
            model = finetune(model, batch, steps=STEPS_PER_BATCH, lr=lr)
    deltas = [(p.detach() - pre[n]).norm().item() for n, p in model.named_parameters()
              if n in pre]
    return traj, eval_ppl(model, user_held), round(sum(deltas) / len(deltas), 4)


def main():
    print('===== 阶段② 变体协议测量（冻结=全部 TF 骨干）=====\n')
    results = {}

    # 预训练 PPL
    for name, v in VARIANTS.items():
        if Path(v['json']).exists():
            r = json.load(open(v['json']))
            results.setdefault(name, {})['pretrain_ppl'] = round(r['best_ppl'], 2)
            results[name]['nan_recoveries'] = r['nan_recoveries']
        else:
            print(f'  [缺] {v["json"]}')

    # 单发
    print('--- 单发冷启动 ---')
    users, tests = make_data()
    for name in VARIANTS:
        if not Path(VARIANTS[name]['ckpt']).exists():
            continue
        imps = []
        for u in range(len(users)):
            m = load_variant(name)
            pb = eval_ppl(m, tests[u])
            m_ft = finetune(m, users[u])
            pa = eval_ppl(m_ft, tests[u])
            imps.append((1 - pa / pb) * 100)
        results[name]['coldstart_pct'] = round(statistics.mean(imps), 1)
        print(f'  {name:9s}: {results[name]["coldstart_pct"]:+.1f}%')
        torch.cuda.empty_cache()

    # 流式
    print('--- 流式 @ {5e-5, 1e-4} ---')
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    for name in VARIANTS:
        if not Path(VARIANTS[name]['ckpt']).exists():
            continue
        m_frozen = load_variant(name)
        frozen = [eval_ppl(m_frozen, b) for b in stream]
        held_frozen = eval_ppl(m_frozen, user_held)
        del m_frozen; torch.cuda.empty_cache()
        for lr in ST_LRS:
            m = load_variant(name)
            traj, held_a, dw = run_stream(m, stream, user_held, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_gain = (1 - held_a / held_frozen) * 100
            results[name][f'stream_lr{lr:g}'] = {
                'online_gain_pct': round(gain, 1),
                'heldout_gain_pct': round(held_gain, 1),
                'cum_dw': dw}
            print(f'  {name:9s} lr={lr:g}: 在线 {gain:+.1f}%  '
                  f'held-out {held_gain:+.1f}%  ΔW {dw}')
            del m; torch.cuda.empty_cache()

    json.dump(results, open(OUT / 'phase2_protocols.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "phase2_protocols.json"}')


if __name__ == '__main__':
    main()
