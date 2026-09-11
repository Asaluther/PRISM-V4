#!/usr/bin/env python3
"""灾难性遗忘基准（GOAL V6#2，首次执行）——权重约束→低遗忘预测的直接检验

科学问题：骨架约束系列发现 PCN 微调时 ΔW ≈ TF/2（第二涌现性质）。
预测：同预算顺序适应下，PCN 对旧任务的遗忘应低于 TF。
若不成立（权重约束≠低遗忘），同样是可入档的否定性结果。

设计（双臂，A→B→A 交替任务流）：
  Arm 1 域内主题切换（multi-interest user）：
    base(TS 因果 checkpoint) → FT(A=奇幻池) → FT(B=校园池) → FT(A 再学习)
  Arm 2 跨域远征（document ingestion）：
    base(TS) → FT(WT 子集) → FT(TS 回归)

公平性（教训 8 对称性）：
  PCN 用 ts_causal_no_gating_s{1-4}（最优 lr 3e-4，PPL≈16.5）
  TF  用 ts_tf_lr1e3_s{1-4}（最优 lr 1e-3，PPL≈14.5-16.5）
  —— 与 fairness_final.json 同源；不用 wt_causal_transformer_s0（lr 次优，PPL 557）。

regime：
  aggressive  —— 统一 lr 5e-4 × 300 步（demo/skeleton 协议原样）。
                 域内 base 上是「重复激进适应的破坏」口径：两种架构都退化，
                 但对称可比；demo 的 +58.5% 增益来自跨域场景，此口径不复现。
  calibrated  —— 各架构各臂用「适应增益最大」的 (lr, steps)（forgetting_calibration.json），
                 干净的「先适应、后遗忘」口径。推荐引用此口径。

协议（两 regime 共有）：冻结 embedding + 前 6 层；AdamW wd 0.01；clip 1.0；
300 步循环 6 个 [1,256] 训练块；eval = exp(min(NLL, 20))。
ΔW 口径与 skeleton 系列一致：layers.* 且层号 ≥6 的参数范数均值。

用法：.venv/Scripts/python.exe forgetting_benchmark.py [aggressive|calibrated]
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import TinyStoriesDataset
from src.data.wikitext import WikiTextDataset

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/mechanism')
SEEDS = [1, 2, 3, 4]
STEPS = 300
FT_LR = 5e-4

KW_A = ['dragon', 'knight', 'castle', 'wizard', 'fairy', 'magic']
KW_B = ['school', 'teacher', 'class', 'homework', 'lesson']
SEQ = 256
N_TRAIN, N_HELD = 6, 4


# ---------------- 数据构造 ----------------

def build_topic_pool(ds, tokenizer, keywords, other_keywords, n_chunks):
    """按关键词从故事流构造主题池，返回 [n_chunks, 257] 的 token 块。
    与另一池严格不相交（同时命中两池关键词的故事丢弃）。"""
    need = n_chunks * (SEQ + 1)
    tokens = []
    for row in ds:
        t = row['text'].lower()
        hit_self = any(k in t for k in keywords)
        hit_other = any(k in t for k in other_keywords)
        if hit_self and not hit_other:
            tokens.extend(tokenizer.encode(row['text']))
            if len(tokens) >= need:
                break
    n = len(tokens) // (SEQ + 1)
    assert n >= n_chunks, f'主题池 token 不足: {len(tokens)} < {need}'
    arr = np.array(tokens[:n_chunks * (SEQ + 1)], dtype=np.int64)
    return arr.reshape(n_chunks, SEQ + 1)


def chunks_to_xy(arr):
    """[n, 257] -> {'x': [n,256], 'y': [n,256]}"""
    return {'x': torch.from_numpy(arr[:, :-1]), 'y': torch.from_numpy(arr[:, 1:])}


def make_eval_sets():
    """四个固定评估集：A_held / B_held / GEN(TS val) / WT_held + 三个训练集"""
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained('gpt2')

    print('  构造主题池（TS train 关键词筛选）...')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    pool_a = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    pool_b = build_topic_pool(ds, tok, KW_B, KW_A, 20)
    # train = 池前 6 块；held = 池末 4 块（中间 10 块留缓冲，降低块边界邻接）
    train_a, held_a = chunks_to_xy(pool_a[:N_TRAIN]), chunks_to_xy(pool_a[-N_HELD:])
    train_b, held_b = chunks_to_xy(pool_b[:N_TRAIN]), chunks_to_xy(pool_b[-N_HELD:])

    print('  GEN：TS validation 固定块...')
    val_ds = TinyStoriesDataset(split='validation', seq_len=SEQ, max_examples=100)
    gen_arr = torch.stack(val_ds.data[10:10 + N_HELD]).numpy()
    gen = chunks_to_xy(gen_arr)

    print('  WT：train 缓存固定块...')
    wt = WikiTextDataset(split='train', seq_len=SEQ)
    train_wt = chunks_to_xy(wt.data[5000:5000 + N_TRAIN].astype(np.int64))
    held_wt = chunks_to_xy(wt.data[6000:6000 + N_HELD].astype(np.int64))

    return {'train_A': train_a, 'held_A': held_a,
            'train_B': train_b, 'held_B': held_b,
            'gen': gen, 'train_WT': train_wt, 'held_WT': held_wt}


# ---------------- 模型与协议 ----------------

def load_model(kind, seed):
    if kind == 'pcn':
        m = PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                     d_gate=64, dropout=0.0, max_seq_len=256, init_mode='fixed')
        ckpt = f'results/ts_causal_no_gating_s{seed}/best_model.pt'
    else:
        m = TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12, n_heads=4,
                             ffn_dim=1024, dropout=0.0, max_seq_len=256,
                             attn_impl='sdpa')
        ckpt = f'results/ts_tf_lr1e3_s{seed}/best_model.pt'
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    return m.to(DEV)


def forward_logits(model, kind, x):
    if kind == 'pcn':
        return model(x, topk=64, no_gating=True)
    return model(x)


@torch.no_grad()
def eval_ppl(model, kind, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = forward_logits(model, kind, x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def finetune(model, kind, train_data, steps=STEPS, lr=FT_LR):
    """冻结微调一段任务，返回 (model, ΔW, 耗时)。协议与 skeleton_probe 一致。"""
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    pre = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    t0 = time.time()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = forward_logits(model, kind, x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    deltas = [(p.detach() - pre[n]).norm().item()
              for n, p in model.named_parameters()
              if n in pre and n.startswith('layers.') and int(n.split('.')[1]) >= 6]
    dw = round(sum(deltas) / len(deltas), 4) if deltas else 0.0
    return model, dw, time.time() - t0


# ---------------- 主流程 ----------------

PHASES = {'arm1': [('P0_base', None),
                   ('P1_ftA', 'train_A'), ('P2_ftB', 'train_B'), ('P3_ftA2', 'train_A')],
          'arm2': [('P0_base', None),
                   ('P1_ftWT', 'train_WT'), ('P2_ftA', 'train_A')]}


def load_calibrated():
    """读校准结果 → {(kind, train_key): (lr, steps)}"""
    cal = json.load(open(OUT / 'forgetting_calibration.json'))
    cfg = {}
    for kind in ('pcn', 'tf'):
        lr_a, s_a = (float(x) for x in cal['arm1'][kind]['best']['config']
                     .replace('lr', '').replace('s', '').split('_'))
        lr_w, s_w = (float(x) for x in cal['arm2'][kind]['best']['config']
                     .replace('lr', '').replace('s', '').split('_'))
        # arm1 三段 + arm2 P2(回归 TS) 用 A 配置；arm2 P1(去 WT) 用 WT 配置
        for ph, tk in PHASES['arm1'] + PHASES['arm2']:
            if tk is None:
                continue
            key = (kind, ph, 'WT' if 'WT' in ph else 'A')
            cfg[key] = (lr_w, int(s_w)) if 'WT' in ph else (lr_a, int(s_a))
    return cfg


def run_arm(kind, seed, arm, data, regime, cal_cfg):
    eval_keys = ['held_A', 'held_B', 'gen', 'held_WT']
    model = load_model(kind, seed)
    out, dws = {}, []
    for ph, train_key in PHASES[arm]:
        if train_key is not None:
            if regime == 'calibrated':
                lr, steps = cal_cfg[(kind, ph, 'WT' if 'WT' in ph else 'A')]
            else:
                lr, steps = FT_LR, STEPS
            model, dw, el = finetune(model, kind, data[train_key], steps=steps, lr=lr)
            dws.append({'phase': ph, 'dw': dw, 'lr': lr, 'steps': steps,
                        'time_s': round(el, 1)})
        out[ph] = {k: round(eval_ppl(model, kind, data[k]), 2) for k in eval_keys}
        print(f'    [{kind} s{seed} {arm}] {ph}: ' +
              '  '.join(f'{k}={out[ph][k]:.1f}' for k in eval_keys))
    return {'ppl': out, 'dw': dws}


def summarize(arm_results, arm):
    """汇总各架构跨 seed 指标（per_seed + mean）"""
    summary = {}
    for kind in ('pcn', 'tf'):
        rs = [arm_results[kind][str(s)] for s in SEEDS]
        agg = {}
        if arm == 'arm1':
            agg['adapt_A_pct'] = [round((1 - r['ppl']['P1_ftA']['held_A'] / r['ppl']['P0_base']['held_A']) * 100, 1) for r in rs]
            agg['forget_A_pct'] = [round((r['ppl']['P2_ftB']['held_A'] / r['ppl']['P1_ftA']['held_A'] - 1) * 100, 1) for r in rs]
            agg['relearn_A_pct'] = [round((1 - r['ppl']['P3_ftA2']['held_A'] / r['ppl']['P2_ftB']['held_A']) * 100, 1) for r in rs]
            agg['learn_B_pct'] = [round((1 - r['ppl']['P2_ftB']['held_B'] / r['ppl']['P0_base']['held_B']) * 100, 1) for r in rs]
            agg['gen_drift_P2_pct'] = [round((r['ppl']['P2_ftB']['gen'] / r['ppl']['P0_base']['gen'] - 1) * 100, 1) for r in rs]
            agg['dw_P1'] = [r['dw'][0]['dw'] for r in rs]
        else:
            agg['adapt_WT_pct'] = [round((1 - r['ppl']['P1_ftWT']['held_WT'] / r['ppl']['P0_base']['held_WT']) * 100, 1) for r in rs]
            agg['ts_forget_pct'] = [round((r['ppl']['P1_ftWT']['gen'] / r['ppl']['P0_base']['gen'] - 1) * 100, 1) for r in rs]
            agg['ts_recover_pct'] = [round((1 - r['ppl']['P2_ftA']['gen'] / r['ppl']['P1_ftWT']['gen']) * 100, 1) for r in rs]
            agg['wt_retention_pct'] = [round((r['ppl']['P2_ftA']['held_WT'] / r['ppl']['P1_ftWT']['held_WT'] - 1) * 100, 1) for r in rs]
            agg['dw_P1'] = [r['dw'][0]['dw'] for r in rs]
        summary[kind] = {k: {'per_seed': v, 'mean': round(sum(v) / len(v), 2)}
                         for k, v in agg.items()}
    return summary


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else 'aggressive'
    assert regime in ('aggressive', 'calibrated'), '用法: forgetting_benchmark.py [aggressive|calibrated]'
    cal_cfg = load_calibrated() if regime == 'calibrated' else None

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f'forgetting_benchmark_{regime}.json'
    print(f'===== 灾难性遗忘基准（{regime}）=====\n')
    print('公平 checkpoint 对: PCN ts_causal_no_gating_s1-4 vs TF ts_tf_lr1e3_s1-4\n')

    data = make_eval_sets()
    results = {'arm1_domain_topic': {'pcn': {}, 'tf': {}},
               'arm2_cross_domain': {'pcn': {}, 'tf': {}}}

    def dump():
        out = {'protocol': {'regime': regime, 'freeze': 'emb+前6层',
                            'train_chunks': N_TRAIN, 'eval_chunks': N_HELD, 'seeds': SEEDS,
                            'checkpoints': {'pcn': 'ts_causal_no_gating_s{1-4}',
                                            'tf': 'ts_tf_lr1e3_s{1-4}（公平版，lr 1e-3）'},
                            'kw_A': KW_A, 'kw_B': KW_B,
                            'note': 'skeleton/demo 系列用的 wt_causal_transformer_s0 为 lr 次优 TF(PPL 557 vs 公平版 215)，本基准改用公平对'},
               'raw': results,
               'date': '2026-09-11'}
        if regime == 'calibrated':
            cal = json.load(open(OUT / 'forgetting_calibration.json'))
            out['protocol']['calibration'] = {a: {k: cal[a][k]['best'] for k in ('pcn', 'tf')}
                                              for a in ('arm1', 'arm2')}
        json.dump(out, open(out_path, 'w'), indent=2, ensure_ascii=False)

    for arm in ('arm1', 'arm2'):
        print(f'\n----- {arm} -----')
        for kind in ('pcn', 'tf'):
            for s in SEEDS:
                rk = 'arm1_domain_topic' if arm == 'arm1' else 'arm2_cross_domain'
                results[rk][kind][str(s)] = run_arm(kind, s, arm, data, regime, cal_cfg)
                dump()  # 增量落盘，崩溃不丢数据

    summary = {'arm1_domain_topic': summarize(results['arm1_domain_topic'], 'arm1'),
               'arm2_cross_domain': summarize(results['arm2_cross_domain'], 'arm2')}

    print('\n===== 判定 =====')
    for name, key in (('域内 forget_A', ('arm1_domain_topic', 'forget_A_pct')),
                      ('跨域 ts_forget', ('arm2_cross_domain', 'ts_forget_pct'))):
        p = summary[key[0]]['pcn'][key[1]]
        t = summary[key[0]]['tf'][key[1]]
        wins = sum(1 for a, b in zip(p['per_seed'], t['per_seed']) if a < b)
        print(f'  {name}: PCN {p["mean"]:+.1f}% vs TF {t["mean"]:+.1f}%  (PCN 更低遗忘胜 {wins}/4 seed)')
    dw_p = summary['arm1_domain_topic']['pcn']['dw_P1']['mean']
    dw_t = summary['arm1_domain_topic']['tf']['dw_P1']['mean']
    print(f'  ΔW(P1): PCN {dw_p:.3f} vs TF {dw_t:.3f}  (TF/PCN = {dw_t/dw_p:.2f}x, 公平 checkpoint)')

    final = json.load(open(out_path))
    final['summary'] = summary
    json.dump(final, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f'\n输出: {out_path}')


if __name__ == '__main__':
    main()
