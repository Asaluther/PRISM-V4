#!/usr/bin/env python3
"""S3/S4: norm_update 与 W_res 假说检验

S3: PCN 去 norm_update（LN before W_up）→ ΔW 是否增大？
S4: PCN 的 W_res 从恒等改为标准残差 x（去掉可学习投影）→ ΔW 是否增大？
S4r: TF 的标准残差 x → W_res 形式（加可学习投影，恒等初始化）→ ΔW 是否减小？
"""
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
def eval_pcn(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x, topk=64, no_gating=True)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


@torch.no_grad()
def eval_tf(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = model(x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def measure_dw(model, pre):
    deltas = []
    for n, p in model.named_parameters():
        if n in pre and n.startswith('layers.'):
            li = int(n.split('.')[1])
            if li >= 6:
                deltas.append((p.detach() - pre[n]).norm().item())
    return round(sum(deltas) / len(deltas), 4) if deltas else 0


def finetune(model, fwd_fn, train_data, steps=STEPS):
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
    return model, measure_dw(model, pre)


# ===== PCN 变体 forward =====

def make_ng_fwd(remove_norm=False, use_pure_residual=False):
    """工厂：生成 no_gating 路径的变体 forward"""
    def fwd(self, x, h_above=None, topk=64, no_feedback=False,
            no_gating=False, return_stats=False):
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

        if remove_norm:
            # S3: 跳过 norm_update，直接 W_up(e + lat)
            update = pc.W_up(e + lat)
        else:
            update = pc.W_up(pc.norm_update(e + lat))

        if use_pure_residual:
            # S4: 标准残差 x（不用 W_res 投影）
            h = F.gelu(update) + x
        else:
            h = F.gelu(update) + pc.W_res(x)
        h = pc.dropout(h)
        return h
    return fwd


# ===== TF 变体 forward（W_res 化残差）=====

def tf_forward_wres(model, x):
    """TF forward，标准残差 x → W_res(x)（可学习投影，恒等初始化）"""
    B, N = x.shape
    h = model.token_emb(x) + model.pos_emb(torch.arange(N, device=x.device))
    h = model.emb_dropout(h)
    for blk in model.layers:
        ln_out = blk.ln1(h)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = blk.attn(ln_out, ln_out, ln_out, attn_mask=mask, need_weights=False)
        h = getattr(blk, 'wres1', torch.eye(h.size(-1), device=h.device))(h) + attn_out
        ln2_out = blk.ln2(h)
        h = getattr(blk, 'wres2', torch.eye(h.size(-1), device=h.device))(h) + blk.ffn(ln2_out)
    h = model.ln_out(h)
    return model.lm_head(h)


@torch.no_grad()
def eval_tf_wres(model, data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(data['x'].size(0)):
        x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
        logits = tf_forward_wres(model, x)
        l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
        tot += l.item(); n += y.numel()
    return math.exp(min(tot / n, 20))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('===== S3/S4: norm_update 与 W_res 假说 =====\n')
    train, heldout = make_data()

    pcn_build = lambda: PCNModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                  n_heads=4, d_gate=64, dropout=0.0,
                                  max_seq_len=256, init_mode='fixed')
    pcn_ckpt = 'results/wt_causal_no_gating_s0/best_model.pt'
    tf_build = lambda: TransformerModel(vocab_size=VOCAB, d_model=256, n_layers=12,
                                         n_heads=4, ffn_dim=1024, dropout=0.0,
                                         max_seq_len=256, attn_impl='mha')
    tf_ckpt = 'results/wt_causal_transformer_s0/best_model.pt'

    results = []

    # ===== 基线复测 =====
    print('--- 基线 ---')
    m = pcn_build()
    m.load_state_dict(torch.load(pcn_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    pb = eval_pcn(m, heldout)
    fwd_normal = lambda mm, xx: mm(xx, topk=64, no_gating=True)
    m_ft, dw = finetune(m, fwd_normal, train)
    pa = eval_pcn(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  PCN 正常          dW={dw:.4f}  PPL {ch:+.1f}%')
    results.append({'tag': 'PCN 正常', 'dw': dw, 'ch': round(ch,1)})

    m = tf_build()
    m.load_state_dict(torch.load(tf_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    pb = eval_tf(m, heldout)
    m_ft, dw = finetune(m, lambda mm, xx: mm(xx), train)
    pa = eval_tf(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  TF 正常           dW={dw:.4f}  PPL {ch:+.1f}%')
    results.append({'tag': 'TF 正常', 'dw': dw, 'ch': round(ch,1)})

    # ===== S3: PCN 去 norm_update =====
    print('\n--- S3: PCN 去 norm_update ---')
    m = pcn_build()
    m.load_state_dict(torch.load(pcn_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    fwd_no_norm = make_ng_fwd(remove_norm=True)
    for blk in m.layers:
        blk.forward = types.MethodType(fwd_no_norm, blk)
    # eval 也要用变体 forward
    @torch.no_grad()
    def eval_variant(model, data):
        model.eval()
        tot, n = 0.0, 0
        for i in range(data['x'].size(0)):
            x, y = data['x'][i:i+1].to(DEV), data['y'][i:i+1].to(DEV)
            logits = model(x, topk=64, no_gating=True)  # 走 patched block forward
            l = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            tot += l.item(); n += y.numel()
        return math.exp(min(tot / n, 20))
    pb = eval_variant(m, heldout)
    m_ft, dw = finetune(m, fwd_normal, train)  # fwd_normal calls model() which uses patched forward
    pa = eval_variant(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  PCN 去norm_update dW={dw:.4f}  PPL {ch:+.1f}%')
    results.append({'tag': 'PCN 去norm', 'dw': dw, 'ch': round(ch,1)})

    # ===== S4: PCN W_res → 纯残差 x =====
    print('\n--- S4: PCN W_res → 纯残差 ---')
    m = pcn_build()
    m.load_state_dict(torch.load(pcn_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    fwd_pure_res = make_ng_fwd(use_pure_residual=True)
    for blk in m.layers:
        blk.forward = types.MethodType(fwd_pure_res, blk)
    pb = eval_variant(m, heldout)
    m_ft, dw = finetune(m, fwd_normal, train)
    pa = eval_variant(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  PCN 纯残差(x)     dW={dw:.4f}  PPL {ch:+.1f}%')
    results.append({'tag': 'PCN 纯残差', 'dw': dw, 'ch': round(ch,1)})

    # ===== S4r: TF 标准残差 → W_res 化 =====
    print('\n--- S4r: TF 残差 W_res 化 ---')
    m = tf_build()
    m.load_state_dict(torch.load(tf_ckpt, map_location=DEV, weights_only=True))
    m = m.to(DEV)
    # 给每块加恒等初始化的 W_res（可学习）
    D = 256
    for blk in m.layers:
        blk.wres1 = torch.nn.Linear(D, D, bias=False)
        torch.nn.init.eye_(blk.wres1.weight)
        blk.wres2 = torch.nn.Linear(D, D, bias=False)
        torch.nn.init.eye_(blk.wres2.weight)
        blk.wres1 = blk.wres1.to(DEV)
        blk.wres2 = blk.wres2.to(DEV)

    pb = eval_tf_wres(m, heldout)

    # finetune with W_res forward
    def finetune_tf_wres(model, train_data, steps=STEPS):
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
            logits = tf_forward_wres(model, x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        model.eval()
        return model, measure_dw(model, pre)

    m_ft, dw = finetune_tf_wres(m, train)
    pa = eval_tf_wres(m_ft, heldout)
    ch = (1 - pa / pb) * 100
    print(f'  TF W_res化残差    dW={dw:.4f}  PPL {ch:+.1f}%')
    results.append({'tag': 'TF W_res化', 'dw': dw, 'ch': round(ch,1)})

    # ===== 判定 =====
    print('\n===== 判定 =====')
    pcn_base = results[0]['dw']
    tf_base = results[1]['dw']
    print(f'  基线:       PCN dW={pcn_base:.4f}  TF dW={tf_base:.4f}  (TF={tf_base/pcn_base:.1f}x)')

    for r in results[2:]:
        ratio = r['dw'] / pcn_base
        verdict = '⬆ 是约束!' if ratio > 1.3 else ('⬇ 有效约束' if ratio < 0.7 else '无变化')
        print(f'  {r["tag"]:<18} dW={r["dw"]:.4f}  ({ratio:.2f}x PCN基线)  {verdict}  PPL {r["ch"]:+.1f}%')

    print('\n全部结果：')
    for r in results:
        print(f'  {r["tag"]:<18} dW={r["dw"]:.4f}  PPL {r["ch"]:+.1f}%')

    json.dump(results, open(OUT / 'skeleton_s3s4.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/skeleton_s3s4.json')


if __name__ == '__main__':
    main()
