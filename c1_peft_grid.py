#!/usr/bin/env python3
"""c1_peft_grid — PEFT 系统网格（评审 W2 规格的响应实验，v6 评审 7.6→8+ 计划 C1）

补齐 peft_3seed 遗留的系统性：此前仅 r8@固定 α16 与 3 个 init。本网格：

  臂 1 LoRA 网格：{TF, HYB} × r ∈ {4,8,16} × α ∈ {16,32} @lr 5e-4
  臂 2 lr 敏感：{TF, HYB} × r8/α16 × lr ∈ {1e-4, 5e-4, 1e-3}
  臂 3 bitfit lr：{TF, HYB} × lr ∈ {1e-3, 5e-4, 1e-4}

口径：单发冷启动（3 用户 × 3 ckpt-seed），LoRA init 固定种子（每格 torch.manual_seed）。
回答：① rank/α 维度是否改变 LoRA-TF 为负的结论；② 最优 lr 是否因 rank/架构而变；
③ bitfit 的 lr 敏感度。

用法：.venv/Scripts/python.exe c1_peft_grid.py   (需 GPU)
输出：results/v7/peft_grid.json（逐格增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from coldstart_090m import VOCAB, DEV, OUT, make_data
from peft_baselines import LoRALinear, scope_blocks, eval_ppl, adapt
from src.models.transformer import TransformerModel
from src.models.hybrid import HybridModel

CKPTS = {'tf':  ['results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
                 'results/wt_200m_tf_lr5e-4_s1/best_model.pt',
                 'results/wt_200m_tf_lr5e-4_s2/best_model.pt'],
         'hyb': ['results/wt_094m_hyb_s0/best_model.pt',
                 'results/wt_094m_hyb_s1/best_model.pt',
                 'results/wt_094m_hyb_s2/best_model.pt']}
INIT_SEED = 202


def load_base(kind, seed):
    if kind == 'hyb':
        m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                        n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                        d_gate=128, dropout=0.0, max_seq_len=256)
    else:
        m = TransformerModel(vocab_size=VOCAB, d_model=512, n_layers=24,
                             n_heads=4, ffn_dim=2048, dropout=0.0,
                             max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load(CKPTS[kind][seed], map_location=DEV,
                                 weights_only=True))
    return m.to(DEV)


def setup_lora(model, kind, rank, alpha):
    """peft_baselines.setup_peft 的参数化版（rank+alpha 可调）"""
    for p in model.parameters():
        p.requires_grad = False

    def wrap(parent):
        n = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.MultiheadAttention):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, LoRALinear(child, r=rank, alpha=alpha))
                n += 1
            else:
                n += wrap(child)
        return n
    wrapped = sum(wrap(b) for b in scope_blocks(model, kind))
    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable, wrapped


def setup_bitfit(model, kind):
    from peft_baselines import setup_peft
    return setup_peft(model, kind, 'bitfit')[0]


def cell(kind, method, rank, alpha, lr, users, tests):
    """一格：3 ckpt-seed × 3 用户，返回逐 seed 均值"""
    seed_avgs = []
    for s in (0, 1, 2):
        imps = []
        for u in range(len(users)):
            m = load_base(kind, s)
            torch.manual_seed(INIT_SEED + s)   # LoRA init 固定种子
            if method == 'lora':
                trainable, _ = setup_lora(m, kind, rank, alpha)
            else:
                trainable = setup_bitfit(m, kind)
            pb = eval_ppl(m, tests[u], kind)
            m = adapt(m, kind, users[u], trainable, 300, lr)
            pa = eval_ppl(m, tests[u], kind)
            imps.append((1 - pa / pb) * 100)
            del m; torch.cuda.empty_cache()
        seed_avgs.append(round(statistics.mean(imps), 1))
    mean = round(statistics.mean(seed_avgs), 1)
    spread = round(max(seed_avgs) - min(seed_avgs), 1)
    return {'per_seed': seed_avgs, 'mean': mean, 'spread': spread}


def main():
    print('===== C1：PEFT 系统网格（单发冷启动，3 seed × 3 用户/格）=====')
    users, tests = make_data()
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(results, open(OUT / 'peft_grid.json', 'w'),
                  indent=2, ensure_ascii=False)

    def key(kind, method, r, a, lr):
        k = f'{kind}/{method}'
        if method == 'lora':
            k += f'/r{r}a{a}@lr{lr:g}'
        else:
            k += f'@lr{lr:g}'
        return k

    # 臂 1：LoRA rank × alpha 网格 @5e-4
    print('\n--- 臂1：LoRA r×α 网格 @5e-4 ---')
    for kind in ('tf', 'hyb'):
        for r in (4, 8, 16):
            for a in (16, 32):
                c = cell(kind, 'lora', r, a, 5e-4, users, tests)
                results[key(kind, 'lora', r, a, 5e-4)] = c
                print(f'  {key(kind, "lora", r, a, 5e-4):26s} '
                      f'{c["mean"]:+7.1f}（逐seed {c["per_seed"]}）', flush=True)
                save()

    # 臂 2：r8/α16 的 lr 敏感（1e-4 已在臂1隐含为对照点，此处补 1e-4 与 1e-3）
    print('\n--- 臂2：r8/α16 lr 敏感 ---')
    for kind in ('tf', 'hyb'):
        for lr in (1e-4, 1e-3):
            c = cell(kind, 'lora', 8, 16, lr, users, tests)
            results[key(kind, 'lora', 8, 16, lr)] = c
            print(f'  {key(kind, "lora", 8, 16, lr):26s} '
                  f'{c["mean"]:+7.1f}（逐seed {c["per_seed"]}）', flush=True)
            save()

    # 臂 3：bitfit lr
    print('\n--- 臂3：bitfit lr 网格 ---')
    for kind in ('tf', 'hyb'):
        for lr in (1e-3, 5e-4, 1e-4):
            c = cell(kind, 'bitfit', None, None, lr, users, tests)
            results[key(kind, 'bitfit', None, None, lr)] = c
            print(f'  {key(kind, "bitfit", None, None, lr):26s} '
                  f'{c["mean"]:+7.1f}（逐seed {c["per_seed"]}）', flush=True)
            save()

    # 速览
    print('\n===== 网格速览 =====')
    tf_lora = {k: v for k, v in results.items()
               if k.startswith('tf/lora')}
    best_tf = max(tf_lora.items(), key=lambda kv: kv[1]['mean'])
    print(f'TF-LoRA 最优格: {best_tf[0]} = {best_tf[1]["mean"]:+.1f}%')
    hyb_lora = {k: v for k, v in results.items() if k.startswith('hyb/lora')}
    best_hyb = max(hyb_lora.items(), key=lambda kv: kv[1]['mean'])
    print(f'HYB-LoRA 最优格: {best_hyb[0]} = {best_hyb[1]["mean"]:+.1f}%')
    save()
    print(f'\n输出: {OUT / "peft_grid.json"}')


if __name__ == '__main__':
    main()
