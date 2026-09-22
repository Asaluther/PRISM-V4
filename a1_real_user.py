#!/usr/bin/env python3
"""a1_real_user — 真实用户数据 + 结构化噪声域双臂适应实验

两臂：
  R（真实用户）：用户本人手写日记（611 字中文 → 1311 GPT-2 tokens → 5 序列）
    —— 预训练域（英文 WT/TS）与用户域（中文个人日记）差异极大
  S（结构化噪声）：程序生成的 LLM 运维笔记（1.25M 字 → 1.85M tokens）
    —— 中文+Markdown+YAML 混合体，模板化但结构复杂

协议：与主线同构——冻结 TF 骨干，PCN 头 50 步 @5e-4（step50 预算）
  R 臂：4 训练序列 + 1 测试序列（数据量受限，协议自动适配）
  S 臂：6 训练序列 + 2 测试序列（对齐主线协议）

测量：冻结 PPL / 适应后 PPL / 改善幅度 / 适应前后生成样本

用法：.venv/Scripts/python.exe a1_real_user.py   (纯 CPU)
输出：results/v7/a1_real_user.json
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json, math, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.models.hybrid import HybridModel

VOCAB = 50257
CKPT = 'results/wt_096m_hyb_h2e-4_s0/best_model.pt'
KW = dict(topk=128, no_gating=True)
TOK_DIR = str(Path(__file__).parent / 'tokenizer')
OUT = Path('results/v7')
LR = 5e-4
STEPS = 50


def load():
    m = HybridModel(vocab_size=VOCAB, d_model=512, n_tf_layers=12,
                    n_pcn_layers=12, n_heads=4, ffn_dim=2048,
                    d_gate=128, dropout=0.0, max_seq_len=256)
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    return m.eval()


def get_tok():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(TOK_DIR)


def text_to_seqs(text, tok, n_train, n_test):
    """文本 → n_train 个训练序列 + n_test 个测试序列（连续切块）"""
    ids = tok.encode(text)
    seqs = []
    for i in range(0, len(ids) - 256, 256):
        chunk = ids[i:i+257]
        seqs.append({'x': torch.tensor([chunk[:-1]]),
                     'y': torch.tensor([chunk[1:]])})
    train = seqs[:n_train]
    test = seqs[n_train:n_train + n_test]
    return train, test


@torch.no_grad()
def eval_ppl(model, seq):
    logits = model(seq['x'], **KW)
    l = F.cross_entropy(logits.reshape(-1, VOCAB),
                        seq['y'].reshape(-1), reduction='sum')
    return math.exp(min(l.item() / seq['y'].numel(), 20))


def adapt(model, train_seqs):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for blk in model.layers:
        for p in blk.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        seq = train_seqs[step % len(train_seqs)]
        logits = model(seq['x'], **KW)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               seq['y'].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def generate(model, tok, prompt, max_new=40, temperature=0.8, seed=42):
    model.eval()
    torch.manual_seed(seed)
    ids = tok.encode(prompt, return_tensors='pt')
    for _ in range(max_new):
        logits = model(ids[:, -256:], **KW)
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0, ids.shape[1] - max_new:])


def run_arm(tag, text, tok, n_train, n_test, gen_prompt):
    print(f'\n--- {tag} ---')
    train, test = text_to_seqs(text, tok, n_train, n_test)
    print(f'  序列: {len(train)} 训练 + {len(test)} 测试')

    m = load()
    ppls_before = [eval_ppl(m, t) for t in test]
    pb = statistics.mean(ppls_before)

    m = adapt(m, train)
    ppls_after = [eval_ppl(m, t) for t in test]
    pa = statistics.mean(ppls_after)
    imp = (1 - pa / pb) * 100

    gen_before = generate(load(), tok, gen_prompt, seed=7)
    gen_after = generate(m, tok, gen_prompt, seed=7)

    rec = {
        'n_train': len(train), 'n_test': len(test),
        'frozen_ppl': round(pb, 1),
        'adapted_ppl': round(pa, 1),
        'improvement_pct': round(imp, 1),
        'per_test_before': [round(p, 1) for p in ppls_before],
        'per_test_after': [round(p, 1) for p in ppls_after],
        'gen_before': gen_before[:200],
        'gen_after': gen_after[:200]}
    print(f'  冻结 PPL: {pb:.1f} → 适应后: {pa:.1f} ({imp:+.1f}%)')
    print(f'  生成(适应前): {gen_before[:80]!r}')
    print(f'  生成(适应后): {gen_after[:80]!r}')
    del m
    return rec


def main():
    print('===== A1：真实用户 + 结构化噪声双臂适应实验 =====')
    tok = get_tok()
    results = {'model': 'h2e4 (终端默认)', 'steps': STEPS, 'lr': LR}

    # R 臂：真实日记
    diary = Path('real_user_diary.txt').read_text(encoding='utf-8')
    results['R_real_diary'] = run_arm(
        'R 真实用户日记', diary, tok, n_train=4, n_test=1,
        gen_prompt='2026年9月23日 星期三')

    # S 臂：结构化噪声
    corpus = Path(r'G:\corpus_gen\corpus_2p5mb.md').read_text(encoding='utf-8')
    results['S_structured_noise'] = run_arm(
        'S 结构化噪声（LLM 运维笔记）', corpus, tok, n_train=6, n_test=2,
        gen_prompt='---\ntitle: 训练稳定性：量化校准')

    # 对照：TinyStories 模拟用户基线
    from src.data.tinystories import TinyStoriesDataset
    import numpy as np
    ds = TinyStoriesDataset(split='validation', seq_len=256, max_examples=100)
    rows = np.stack([np.asarray(r) for r in ds.data[:8]])
    d = torch.from_numpy(rows.astype('int64'))
    ts_train = [{'x': d[i:i+1, :-1], 'y': d[i:i+1, 1:]} for i in range(6)]
    ts_test = [{'x': d[6:8, :-1], 'y': d[6:8, 1:]}]
    m = load()
    pb = statistics.mean([eval_ppl(m, t) for t in ts_test])
    m = adapt(m, ts_train)
    pa = statistics.mean([eval_ppl(m, t) for t in ts_test])
    imp = (1 - pa / pb) * 100
    results['baseline_tinystories'] = {
        'frozen_ppl': round(pb, 1), 'adapted_ppl': round(pa, 1),
        'improvement_pct': round(imp, 1)}
    print(f'\n--- 对照 TinyStories 模拟 ---')
    print(f'  冻结 {pb:.1f} → 适应后 {pa:.1f} ({imp:+.1f}%)')

    # 汇总
    print('\n===== 汇总 =====')
    for k in ('R_real_diary', 'S_structured_noise', 'baseline_tinystories'):
        r = results[k]
        print(f"  {k:24s}: {r['frozen_ppl']:.0f} → {r['adapted_ppl']:.0f} "
              f"({r['improvement_pct']:+.1f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT / 'a1_real_user.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'输出: {OUT / "a1_real_user.json"}')


if __name__ == '__main__':
    main()
