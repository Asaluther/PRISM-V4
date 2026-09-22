#!/usr/bin/env python3
"""adapt500 — 500M 检查点的适应协议复测（+ 330M HYB 延伸）

主体：TF@2.5e-4 @536.9M（唯一存活的 500M 检查点——从启智回传）
  冷启动：3 用户 × 300 步 @5e-4（底 12 冻顶 12 训，对齐主线 TF 协议）
  流式：8 批 × 60 步 @ {5e-5, 1e-4}
  bitfit 臂：@1e-3（TF 系最强 PEFT 配置）

330M HYB 对照：冷启动 + 流式（检查点在本地，可立即跑）

预注册判读：
  A. TF@2.5e-4 500M 冷启动灾难（TF 同型）→ 适应脆弱性是架构属性，不随规模消失
  B. TF@2.5e-4 500M bitfit 恢复 → 方法×规模交互正常
  C. 330M HYB 适应优势保持 → 与已有 adapt330.json 对照

用法：.venv/Scripts/python.exe adapt500.py   (需 GPU)
输出：results/v7/adapt500.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import VOCAB, DEV, make_data

KW_HYB = dict(topk=128, no_feedback=False, no_gating=True,
              feedback_mode='prev_layer', n_pass=1)
FREEZE_LAYERS = 12
OUT = Path('results/v7')

# 500M TF 用 transformers 原生 GPT2（外部检查点格式）
# 330M HYB 用项目内 HybridModel
from src.models.hybrid import HybridModel
from src.models.transformer import TransformerModel


def load_330_hyb():
    m = HybridModel(vocab_size=VOCAB, d_model=1024, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=8, ffn_dim=4096,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(
        'results/wt_330m_hyb_ng_lr5e-4_s0/best_model.pt',
        map_location=DEV, weights_only=True))
    return m.to(DEV)


def load_500_tf():
    """TF 500M 用项目内 TransformerModel（d1280 与训练配置一致）"""
    m = TransformerModel(vocab_size=VOCAB, d_model=1280, n_layers=24,
                         n_heads=10, ffn_dim=5120, dropout=0.0,
                         max_seq_len=256, attn_impl='sdpa')
    m.load_state_dict(torch.load(
        'results/wt_553m_tf_lr2.5e-4_s0/best_model.pt',
        map_location=DEV, weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def ev(m, d, kw=None):
    tot, n = 0.0, 0
    m.eval()
    for i in range(d['x'].size(0)):
        x, y = d['x'][i:i+1].to(DEV), d['y'][i:i+1].to(DEV)
        logits = m(x, **kw) if kw else m(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB),
                            y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def setup_trainable(m, mode, is_hyb):
    """mode: 'full_top12' | 'bitfit'"""
    for p in m.parameters():
        p.requires_grad = False
    if is_hyb:
        for blk in m.layers:  # TF 骨干全冻
            for p in blk.parameters():
                p.requires_grad = False
        if mode == 'full_top12':
            for p in m.parameters():
                if p.requires_grad or True:
                    pass
        trainable = [p for p in m.parameters() if not p.requires_grad]
        # HYB：PCN 头全部可训练
        for p in m.pcn_blocks.parameters():
            p.requires_grad = True
        trainable = [p for p in m.parameters() if p.requires_grad]
    else:
        if mode == 'full_top12':
            for blk in m.layers[FREEZE_LAYERS:]:
                for p in blk.parameters():
                    p.requires_grad = True
        elif mode == 'bitfit':
            import torch.nn as nn
            for blk in m.layers[FREEZE_LAYERS:]:
                for nm, p in blk.named_parameters():
                    if nm.endswith('bias') or (nm.endswith('.weight') and
                            isinstance(dict(blk.named_modules()).get(
                                nm.rsplit('.', 1)[0]), nn.LayerNorm)):
                        p.requires_grad = True
            for nm, p in m.ln_out.named_parameters():
                p.requires_grad = True
    return [p for p in m.parameters() if p.requires_grad]


def adapt(m, data, mode, is_hyb, steps, lr):
    trainable = setup_trainable(m, mode, is_hyb)
    if not trainable:
        return m
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    kw = KW_HYB if is_hyb else None
    xs, ys = data['x'].to(DEV), data['y'].to(DEV)
    m.train()
    for step in range(steps):
        ci = step % xs.size(0)
        logits = m(xs[ci:ci+1], **kw) if kw else m(xs[ci:ci+1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    m.eval()
    return m


def main():
    print('===== 500M/330M 适应协议复测 =====')
    users, tests = make_data()
    results = {}

    # 流式数据
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained(str(Path(__file__).parent / 'tokenizer'))
    ds = load_dataset('roneneldan/TinyStories', split='train')
    from forgetting_benchmark import build_topic_pool, chunks_to_xy, KW_A, KW_B
    pool = build_topic_pool(ds, tok, KW_A, KW_B, 20)
    stream = [chunks_to_xy(pool[i:i+1]) for i in range(8)]
    user_held = chunks_to_xy(pool[-4:])

    # ========== 330M HYB（本地检查点，立即可跑）==========
    print('\n--- 330M HYB 冷启动 ---')
    for mode, lr in [('full', 5e-4)]:
        imps = []
        for u in range(3):
            m = load_330_hyb()
            pb = ev(m, tests[u], KW_HYB)
            m = adapt(m, users[u], mode, True, 300, lr)
            pa = ev(m, tests[u], KW_HYB)
            imps.append(round((1 - pa / pb) * 100, 1))
            del m; torch.cuda.empty_cache()
        results['hyb330_coldstart'] = {'per_user': imps,
                                       'avg': round(statistics.mean(imps), 1)}
        print(f'  {mode}@{lr:g}: {imps} -> {statistics.mean(imps):+.1f}%')

    print('\n--- 330M HYB 流式 ---')
    m_frozen = load_330_hyb()
    frozen = [ev(m_frozen, b, KW_HYB) for b in stream]
    held_frozen = ev(m_frozen, user_held, KW_HYB)
    del m_frozen; torch.cuda.empty_cache()
    for lr in (5e-5, 1e-4):
        m = load_330_hyb()
        traj = []
        for i, batch in enumerate(stream):
            traj.append(round(ev(m, batch, KW_HYB), 2))
            if i < len(stream) - 1:
                m = adapt(m, batch, 'full', True, 60, lr)
        gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
        held_a = ev(m, user_held, KW_HYB)
        held_gain = (1 - held_a / held_frozen) * 100
        results[f'hyb330_stream_lr{lr:g}'] = {
            'online_gain_pct': round(gain, 1),
            'heldout_gain_pct': round(held_gain, 1)}
        print(f'  lr={lr:g}: 在线 {gain:+.1f}%  held {held_gain:+.1f}%')
        del m; torch.cuda.empty_cache()

    # ========== 500M TF@2.5e-4（需检查点就位）==========
    ckpt = Path('results/wt_553m_tf_lr2.5e-4_s0/best_model.pt')
    if not ckpt.exists():
        print('\n[缺] 500M TF 检查点未找到——请从启智下载放入:')
        print(f'  {ckpt}')
        print('  跳过 500M 部分，仅输出 330M 结果')
    else:
        print('\n--- 500M TF@2.5e-4 冷启动（full-FT + bitfit）---')
        for mode, lr in [('full_top12', 5e-4), ('bitfit', 1e-3)]:
            imps = []
            for u in range(3):
                m = load_500_tf()
                pb = ev(m, tests[u])
                m = adapt(m, users[u], mode, False, 300, lr)
                pa = ev(m, tests[u])
                imps.append(round((1 - pa / pb) * 100, 1))
                del m; torch.cuda.empty_cache()
            results[f'tf500_{mode}'] = {'per_user': imps,
                                         'avg': round(statistics.mean(imps), 1)}
            print(f'  {mode}@{lr:g}: {imps} -> {statistics.mean(imps):+.1f}%')

        print('\n--- 500M TF@2.5e-4 流式 ---')
        m_frozen = load_500_tf()
        frozen = [ev(m_frozen, b) for b in stream]
        held_frozen = ev(m_frozen, user_held)
        del m_frozen; torch.cuda.empty_cache()
        for lr in (5e-5, 1e-4):
            m = load_500_tf()
            traj = []
            for i, batch in enumerate(stream):
                traj.append(round(ev(m, batch), 2))
                if i < len(stream) - 1:
                    m = adapt(m, batch, 'full_top12', False, 60, lr)
            gain = (1 - sum(traj[1:]) / sum(frozen[1:])) * 100
            held_a = ev(m, user_held)
            held_gain = (1 - held_a / held_frozen) * 100
            results[f'tf500_stream_lr{lr:g}'] = {
                'online_gain_pct': round(gain, 1),
                'heldout_gain_pct': round(held_gain, 1)}
            print(f'  lr={lr:g}: 在线 {gain:+.1f}%  held {held_gain:+.1f}%')
            del m; torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'adapt500.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "adapt500.json"}')


if __name__ == '__main__':
    main()
