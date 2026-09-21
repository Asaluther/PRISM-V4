#!/usr/bin/env python3
"""c2_mechanism — C2a 适应动力学纵向日志 + C2b 梯度阻尼干预（评审 W4/Q2）

C2a（动力学）：h2e4 / 纯PCN / TF 三架构冷启动 300 步全程，每 10 步记录：
  - 分组梯度范数（TF 骨干 vs 可训练头）—— backward 后、step 前读
  - 分组 ΔW（相对初始权重）
  - PCN 逐层误差范数 ‖e‖（blk.last_e，模型已存）
  - train / test / general(WT) 三域 PPL
回答评审「误差正反馈是否有微观证据」：预期 PCN 头梯度范数在过拟合相增长。

C2b（阻尼干预）：h2e4 冷启动，头梯度钳制 head_clip ∈ {1.0(基线), 0.3, 0.1}
（全局 clip 恒 1.0，另对头参数额外钳制）。若阻尼降低遗忘（gen 恶化变小）而
保留适应（test 改善）→ 误差正反馈的因果证据 + 终端「阻尼预算旋钮」新选项。

用法：.venv/Scripts/python.exe c2_mechanism.py   (需 GPU)
输出：results/v7/adapt_dynamics.json（增量保存）
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from coldstart_090m import VOCAB, DEV, OUT, make_data, build_model
from src.data.wikitext import WikiTextDataset
import numpy as np

PCN_KW = dict(topk=128, no_feedback=False, no_gating=True,
              feedback_mode='prev_layer', n_pass=1)
FREEZE_LAYERS = 12
OUT = Path('results/v7')


def load(kind):
    CKPT = {'pcn': 'results/wt_090m_pcn_lr1e-4_ckpt/best_model.pt',
            'tf':  'results/wt_200m_tf_lr5e-4_ckpt/best_model.pt',
            'hyb': 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'}
    m = build_model(kind)
    m.load_state_dict(torch.load(CKPT[kind], map_location=DEV,
                                 weights_only=True))
    return m.to(DEV)


@torch.no_grad()
def ev(m, d, kw):
    tot, n = 0.0, 0
    m.eval()
    for i in range(d['x'].size(0)):
        x, y = d['x'][i:i+1].to(DEV), d['y'][i:i+1].to(DEV)
        logits = m(x, **kw) if kw else m(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                            reduction='sum')
        tot += l.item(); n += y.numel()
    m.train()
    return math.exp(min(tot / n, 20))


def freeze(m, kind):
    m.token_emb.weight.requires_grad = False
    m.pos_emb.weight.requires_grad = False
    nf = m.layers if kind == 'hyb' else m.layers[:FREEZE_LAYERS]
    for blk in nf:
        for p in blk.parameters():
            p.requires_grad = False
    return [p for p in m.parameters() if p.requires_grad]


def group_grad_norm(m, kind):
    """TF 骨干（冻结侧≈0）与可训练头的梯度范数"""
    head, trunk = 0.0, 0.0
    for n_, p in m.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().norm().item()
        if n_.startswith(('layers.',)) and kind != 'hyb':
            trunk += g
        elif n_.startswith('layers.') and kind == 'hyb':
            trunk += g          # hyb 的 layers=TF 骨干（冻结，应≈0）
        else:
            head += g           # pcn_blocks / 顶12层(tf)
    return head, trunk


def run_dynamics(kind, user, test, gen):
    kw = PCN_KW if kind in ('pcn', 'hyb') else None
    m = load(kind)
    trainable = freeze(m, kind)
    pre = {n_: p.detach().clone() for n_, p in m.named_parameters()
           if p.requires_grad}
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    xs, ys = user['x'].to(DEV), user['y'].to(DEV)

    curve = []
    for step in range(1, 301):
        ci = (step - 1) % xs.size(0)
        logits = m(xs[ci:ci+1], **kw) if kw else m(xs[ci:ci+1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), ys[ci:ci+1].reshape(-1))
        opt.zero_grad(); loss.backward()
        # 步前记录梯度范数
        if step % 10 == 0:
            head_g, trunk_g = group_grad_norm(m, kind)
            dw = sum((p.detach() - pre[n_]).norm().item()
                     for n_, p in m.named_parameters() if n_ in pre)
            e_norms = []
            blocks = getattr(m, 'pcn_blocks', None) or getattr(m, 'layers', [])
            for i in range(0, len(blocks), 3):
                blk = blocks[i]
                layer = blk.pcn_layer if hasattr(blk, 'pcn_layer') else blk
                if hasattr(layer, 'last_e') and layer.last_e is not None:
                    e_norms.append(round(layer.last_e.norm().item(), 2))
            curve.append({'step': step,
                          'loss': round(loss.item(), 4),
                          'head_grad': round(head_g, 4),
                          'trunk_grad': round(trunk_g, 5),
                          'dw': round(dw, 3),
                          'e_norms_sample': e_norms})
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 10 == 0:
            curve[-1]['train_ppl'] = round(ev(m, user, kw), 1)
            curve[-1]['test_ppl'] = round(ev(m, test, kw), 1)
            curve[-1]['gen_ppl'] = round(ev(m, gen, kw), 1)
    del m; torch.cuda.empty_cache()
    return curve


def run_damping(user, test, gen):
    """C2b：头梯度额外钳制 ∈ {1.0, 0.3, 0.1}（全局 clip 后追加）"""
    out = {}
    for head_clip in (1.0, 0.3, 0.1):
        m = load('hyb')
        trainable = freeze(m, 'hyb')
        head_params = [p for n_, p in m.named_parameters()
                       if p.requires_grad and n_.startswith('pcn_blocks')]
        opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
        xs, ys = user['x'].to(DEV), user['y'].to(DEV)
        pb = ev(m, test, PCN_KW); gb = ev(m, gen, PCN_KW)
        m.train()
        for step in range(1, 301):
            ci = (step - 1) % xs.size(0)
            logits = m(xs[ci:ci+1], **PCN_KW)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                                   ys[ci:ci+1].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)   # 全局
            if head_clip < 1.0:                               # 头额外阻尼
                torch.nn.utils.clip_grad_norm_(head_params, head_clip)
            opt.step()
        m.eval()
        pa = ev(m, test, PCN_KW); ga = ev(m, gen, PCN_KW)
        out[f'clip{head_clip:g}'] = {
            'test_gain_pct': round((1 - pa / pb) * 100, 1),
            'gen_degr_pct': round((ga / gb - 1) * 100, 1)}
        print(f'  头钳 {head_clip:g}: test {out[f"clip{head_clip:g}"]["test_gain_pct"]:+.1f}% '
              f'| gen 恶化 {out[f"clip{head_clip:g}"]["gen_degr_pct"]:+.1f}%', flush=True)
        del m; torch.cuda.empty_cache()
    return out


def main():
    print('===== C2：适应动力学 + 梯度阻尼干预 =====')
    users, tests = make_data()
    wt = WikiTextDataset(split='validation', seq_len=256)
    gen = {'x': torch.from_numpy(wt.data[:4, :-1].astype('int64')),
           'y': torch.from_numpy(wt.data[:4, 1:].astype('int64'))}
    results = {}
    OUT.mkdir(parents=True, exist_ok=True)

    # C2a：三架构 × user0
    for tag, kind in (('h2e4', 'hyb'), ('pcn', 'pcn'), ('tf', 'tf')):
        print(f'\n--- C2a 动力学: {tag} ---', flush=True)
        curve = run_dynamics(kind, users[0], tests[0], gen)
        results[f'dynamics_{tag}'] = curve
        head0, headN = curve[0]['head_grad'], curve[-1]['head_grad']
        print(f'  头梯度范数: step10 {head0} -> step300 {headN} '
              f'({headN/head0:.2f}×)', flush=True)
        print(f'  gen PPL: {curve[0]["gen_ppl"]} -> {curve[-1]["gen_ppl"]}',
              flush=True)
        json.dump(results, open(OUT / 'adapt_dynamics.json', 'w'),
                  indent=2, ensure_ascii=False)

    # C2b：阻尼（user0，h2e4）
    print('\n--- C2b 阻尼干预: h2e4 ---', flush=True)
    results['damping_h2e4'] = run_damping(users[0], tests[0], gen)
    json.dump(results, open(OUT / 'adapt_dynamics.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT / "adapt_dynamics.json"}')


if __name__ == '__main__':
    main()
