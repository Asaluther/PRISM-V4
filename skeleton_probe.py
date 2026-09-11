#!/usr/bin/env python3
"""骨架约束根源探挖：GELU 门控假说双向检验"""
import sys, json, math, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders

DEV = torch.device('cuda')
VOCAB = 50257
OUT = Path('results/mechanism')
STEPS = 300


def make_data():
    train_loader, _, tok = get_dataloaders(seq_len=256, batch_size=32,
                                            num_workers=0, max_train=50000, max_val=100)
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 2:
            break
        batches.append((x, y))
    return {'x': batches[0][0][:6], 'y': batches[0][1][:6]}, \
           {'x': batches[1][0][:4], 'y': batches[1][1][:4]}


@torch.no_grad()
def eval_ppl_pcn(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, topk=64, no_gating=True)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


@torch.no_grad()
def eval_ppl_tf(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def measure_deltas(model, pre):
    deltas = []
    for n, p in model.named_parameters():
        if n in pre and n.startswith('layers.'):
            li = int(n.split('.')[1])
            if li >= 6:
                deltas.append((p.detach() - pre[n]).norm().item())
    return round(sum(deltas) / len(deltas), 4) if deltas else 0


def finetune_pcn(model, train_data, steps=STEPS):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    pre = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = model(x, topk=64, no_gating=True)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    model.eval()
    return model, measure_deltas(model, pre)


def finetune_tf(model, train_data, fwd_fn, steps=STEPS):
    model.token_emb.weight.requires_grad = False
    model.pos_emb.weight.requires_grad = False
    for i, blk in enumerate(model.layers):
        if i < 6:
            for p in blk.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    pre = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    xs, ys = train_data['x'].to(DEV), train_data['y'].to(DEV)
    model.train()
    for step in range(steps):
        ci = step % xs.size(0)
        x, y = xs[ci:ci+1], ys[ci:ci+1]
        logits = fwd_fn(model, x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    model.eval()
    return model, measure_deltas(model, pre)


def ng_no_gelu_forward(self, x, h_above=None, topk=64, no_feedback=False,
                       no_gating=False, return_stats=False):
    """no_gating 路径，GELU 替换为恒等"""
    pc = self.pcn_layer
    u = pc.W_bu(pc.norm_bu(x))
    if h_above is not None:
        p = pc.W_td(h_above)
    else:
        p = torch.zeros_like(u)
    e = (u - p).clamp(-10.0, 10.0)
    pc.last_e = e.detach()
    pc.last_h_above = h_above.detach() if h_above is not None else None
    pc.last_x = x.detach()
    mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device,
                                  dtype=torch.bool), diagonal=1)
    attn_out, _ = self.attn(e, e, e, attn_mask=mask, need_weights=False)
    lat = self.attn_out(attn_out)
    update = pc.W_up(pc.norm_update(e + lat))
    h = update + pc.W_res(x)  # NO GELU
    h = pc.dropout(h)
    return h


def tf_forward_gelu(model, x):
    """TF forward，残差更新加 GELU 门控"""
    B, N = x.shape
    h = model.token_emb(x) + model.pos_emb(torch.arange(N, device=x.device))
    h = model.emb_dropout(h)
    for blk in model.layers:
        ln_out = blk.ln1(h)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = blk.attn(ln_out, ln_out, ln_out, attn_mask=mask, need_weights=False)
        h = h + F.gelu(attn_out)  # GELU gate (attn_out already projected by MHA)
        ln2_out = blk.ln2(h)
        h = h + blk.ffn(ln2_out)
    h = model.ln_out(h)
    return model.lm_head(h)


@torch.no_grad()
def eval_ppl_tf_gelu(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = tf_forward_gelu(model, x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== 骨架约束根源探挖 =====\n')
    train, heldout = make_data()
    results = []

    pcn_build = lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                  n_heads=4, d_gate=64, dropout=0.0,
                                  max_seq_len=256, init_mode='fixed')
    pcn_ckpt = 'results/wt_causal_no_gating_s0/best_model.pt'
    tf_build = lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                         n_heads=4, ffn_dim=1024, dropout=0.0,
                                         max_seq_len=256, attn_impl='mha')
    tf_ckpt = 'results/wt_causal_transformer_s0/best_model.pt'

    # 基线
    print('--- 基线 ---')
    m = pcn_build()
    m.load_state_dict(torch.load(pcn_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    pb = eval_ppl_pcn(m, heldout)
    m_ft, dw = finetune_pcn(m, train)
    pa = eval_ppl_pcn(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  PCN 正常        dW={dw:.4f}  PPL {pb:.0f}>{pa:.0f} ({ch:+.1f}%)')
    results.append({'tag': 'PCN 正常', 'dw': dw, 'ppl_b': round(pb,1),
                    'ppl_a': round(pa,1), 'ch': round(ch,1)})

    m = tf_build()
    m.load_state_dict(torch.load(tf_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    pb = eval_ppl_tf(m, heldout)
    m_ft, dw = finetune_tf(m, train, lambda mm, xx: mm(xx))
    pa = eval_ppl_tf(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  TF 正常         dW={dw:.4f}  PPL {pb:.0f}>{pa:.0f} ({ch:+.1f}%)')
    results.append({'tag': 'TF 正常', 'dw': dw, 'ppl_b': round(pb,1),
                    'ppl_a': round(pa,1), 'ch': round(ch,1)})

    # S1a: PCN 去 GELU
    print('\n--- S1a: PCN 去 GELU ---')
    m = pcn_build()
    m.load_state_dict(torch.load(pcn_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    for blk in m.layers:
        blk.forward = types.MethodType(ng_no_gelu_forward, blk)

    pb = eval_ppl_pcn(m, heldout)
    m_ft, dw = finetune_pcn(m, train)
    pa = eval_ppl_pcn(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  PCN 去GELU      dW={dw:.4f}  PPL {pb:.0f}>{pa:.0f} ({ch:+.1f}%)')
    results.append({'tag': 'PCN 去GELU', 'dw': dw, 'ppl_b': round(pb,1),
                    'ppl_a': round(pa,1), 'ch': round(ch,1)})

    # S1b: TF + GELU
    print('\n--- S1b: TF + GELU 门控 ---')
    m = tf_build()
    m.load_state_dict(torch.load(tf_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    pb = eval_ppl_tf_gelu(m, heldout)
    m_ft, dw = finetune_tf(m, train, tf_forward_gelu)
    pa = eval_ppl_tf_gelu(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  TF +GELU门控    dW={dw:.4f}  PPL {pb:.0f}>{pa:.0f} ({ch:+.1f}%)')
    results.append({'tag': 'TF +GELU门控', 'dw': dw, 'ppl_b': round(pb,1),
                    'ppl_a': round(pa,1), 'ch': round(ch,1)})

    # 判定
    print('\n===== 判定 =====')
    pcn_base = results[0]['dw']
    tf_base = results[1]['dw']
    pcn_ng = results[2]['dw']
    tf_g = results[3]['dw']
    print(f'  基线:     PCN dW={pcn_base:.4f}  TF dW={tf_base:.4f}  (TF={tf_base/pcn_base:.1f}x PCN)')
    print(f'  S1a PCN去GELU: dW={pcn_ng:.4f}  ({pcn_ng/pcn_base:.2f}x)  '
          + ('GELU是约束 YES' if pcn_ng > pcn_base * 1.3 else 'GELU非约束'))
    print(f'  S1b TF+GELU:  dW={tf_g:.4f}  ({tf_g/tf_base:.2f}x)   '
          + ('GELU在TF也约束 YES' if tf_g < tf_base * 0.7 else 'GELU对TF无效'))

    print('\n泛化效应：')
    for r in results:
        print(f'  {r["tag"]:<16} PPL {r["ch"]:+.1f}%')

    json.dump(results, open(OUT / 'skeleton_constraint.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/skeleton_constraint.json')


if __name__ == '__main__':
    main()
