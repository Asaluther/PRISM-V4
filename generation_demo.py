#!/usr/bin/env python3
"""V6 Demo D3 — 文本生成对比：微调前 vs 微调后的风格变化

用用户 1 的 PCN checkpoint（微调前后各生成一段），展示个性化效果。
同时展示 Transformer 微调前后的对比（预期变化较小或退化）。
"""
import sys, json, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders
from transformers import GPT2TokenizerFast

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/demo')


def make_user_data(num_users=1, tokens_per_user=1536):
    train_loader, _, tok = get_dataloaders(seq_len=256, batch_size=32,
                                            num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= num_users * 2 + 1:
            break
        batches.append((x, y))

    users = []
    for u in range(num_users):
        x, y = batches[u * 2]
        xs = [x[i:i+1] for i in range(min(6, x.size(0)))]
        ys = [y[i:i+1] for i in range(min(6, x.size(0)))]
        users.append({'x': torch.cat(xs), 'y': torch.cat(ys)})

    # prompt：从用户数据的第一条取前 20 个 token 作为生成起点
    prompts = [batches[u * 2][0][0][:20].clone() for u in range(num_users)]
    # 用户原文（用于风格参考）
    originals = []
    for u in range(num_users):
        x = batches[u * 2][0][0]
        originals.append(x)
    return users, prompts, originals, tok


def finetune(model, kind, user_data, steps=300, lr=5e-4):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    xs, ys = user_data['x'].to(DEV), user_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        if kind == 'pcn':
            logits = model(x, topk=64, no_gating=True)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def generate(model, prompt_ids, tok, max_new=80, temperature=0.8, top_k=50, kind='pcn'):
    """从 prompt 生成文本"""
    ids = prompt_ids.to(DEV).unsqueeze(0)  # [1, 20]
    for _ in range(max_new):
        if kind == 'pcn':
            logits = model(ids[:, -256:], topk=64, no_gating=True)
        else:
            logits = model(ids[:, -256:])
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== D3: 文本生成对比 =====\n')

    users, prompts, originals, tok = make_user_data()

    # 用户原文（风格参考）
    print('--- 用户原文（风格参考，前 80 token）---')
    original_text = tok.decode(originals[0][:80].tolist())
    print(f'  {original_text[:200]}...\n')

    results = {}

    for tag, build, kind, ckpt in (
        ('PCN', lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                  n_heads=4, d_gate=64, dropout=0.0,
                                  max_seq_len=256, init_mode='fixed'),
         'pcn', 'results/wt_causal_no_gating_s0/best_model.pt'),
        ('TF', lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                         n_heads=4, ffn_dim=1024, dropout=0.0,
                                         max_seq_len=256, attn_impl='mha'),
         'tf', 'results/wt_causal_transformer_s0/best_model.pt'),
    ):
        print(f'--- {tag} ---')
        model = build()
        model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
        model = model.to(DEV).eval()

        # 微调前生成
        torch.manual_seed(42)
        text_before = generate(model, prompts[0], tok, kind=kind)
        print(f'  [微调前] {text_before[:150]}...')

        # 微调
        model_ft = copy.deepcopy(model)
        model_ft = finetune(model_ft, kind, users[0])

        # 微调后生成（同 seed 同 prompt）
        torch.manual_seed(42)
        text_after = generate(model_ft, prompts[0], tok, kind=kind)
        print(f'  [微调后] {text_after[:150]}...')
        print()

        results[tag] = {'before': text_before, 'after': text_after}

    results['original'] = original_text
    json.dump(results, open(OUT / 'generation_comparison.json', 'w'),
              indent=2, ensure_ascii=False)
    print(f'输出: {OUT}/generation_comparison.json')


if __name__ == '__main__':
    main()
