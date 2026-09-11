#!/usr/bin/env python3
"""PRISM V6 级别 2 — 主路径参数的局部学习规则（LM 任务，真分水岭）

设计：results/V6_LOCAL_RULE_DESIGN.md §4R
- M0：全 BP（基线；有效性自检 best PPL ≤ 25，否则实验无效）
- M1：W_td + W_bu 局部规则，其余 BP
- M2：M1 + W_up/W_res 局部（主路径全局部；e_{l+1} 用上一步的值——时间局部，
      在线学习语义）
判据链：M1 ≤ M0×1.15（PPL 比）→ M2 ≤ M0×1.3

用法：python local_rule_level2.py --mode M0|M1|M2 [--steps 5000]
GPU 空闲时运行（与曲线队列串行）。
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.models.pcn import PCNModel
from src.data.tinystories import get_dataloaders
from train import get_cosine_schedule

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('results/mechanism')


def local_params(mode):
    if mode == 'M1':
        return ('W_td', 'W_bu')
    if mode == 'M2':
        return ('W_td', 'W_bu', 'W_up', 'W_res')
    return ()


def run(mode, steps=5000, lr=3e-4, lr_loc=3e-4):
    torch.manual_seed(0)
    model = PCNModel(vocab_size=50257, d_model=256, n_layers=12, n_heads=4,
                     d_gate=64, dropout=0.1, max_seq_len=256,
                     init_mode='fixed').to(DEV)
    local_names = local_params(mode)
    bp_params = [p for n_, p in model.named_parameters()
                 if not any(n_.endswith(f'{w}.weight') for w in local_names)]
    opt = AdamW(bp_params, lr=lr, weight_decay=0.1)
    # 复用 train.py 的已验证调度器（自写版曾因 warmup 锚点错误导致基线 52 vs 16.5）
    sched = get_cosine_schedule(opt, 500, steps)

    train_loader, val_loader, _ = get_dataloaders(seq_len=256, batch_size=32,
                                                  num_workers=2, max_train=50000, max_val=5000)
    data_iter = iter(train_loader)
    prev_next_e = [None] * model.n_layers   # 上一步各层 e（时间局部代理）
    best_ppl, curve = float('inf'), []
    t0 = time.time()
    step = 0
    while step < steps:
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            b = next(data_iter)
        x, y = b[0].to(DEV), b[1].to(DEV)
        logits = model(x, topk=64, no_gating=True)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               y[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(bp_params, 1.0)
        opt.step(); sched.step()

        if local_names:
            with torch.no_grad():
                D = model.d_model
                for li, blk in enumerate(model.layers):
                    pc = blk.pcn_layer
                    e = pc.last_e
                    x_l = pc.last_x
                    h_above = pc.last_h_above
                    # M1: W_td/W_bu
                    if h_above is not None:
                        pc.W_td.weight += lr_loc * (e.reshape(-1, D).t() @ h_above.reshape(-1, D)) / e.size(0) / x.size(0)
                    pc.W_bu.weight += lr_loc * (e.reshape(-1, D).t() @ x_l.reshape(-1, D)) / e.size(0) / x.size(0)
                    # M2: W_up/W_res（输出误差用上一步更深层的 e——时间局部）
                    if mode == 'M2' and li + 1 < model.n_layers and prev_next_e[li + 1] is not None:
                        en = prev_next_e[li + 1]
                        pc.W_res.weight += lr_loc * (en.reshape(-1, D).t() @ x_l.reshape(-1, D)) / en.size(0) / x.size(0)
                    prev_next_e[li] = e
        step += 1
        if step % 1000 == 0 or step == steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    vx, vy = vb[0].to(DEV), vb[1].to(DEV)
                    vl = model(vx, topk=64, no_gating=True)
                    tot += F.cross_entropy(vl[:, :-1].reshape(-1, vl.size(-1)),
                                           vy[:, 1:].reshape(-1), reduction='sum').item()
                    n += vy[:, 1:].numel()
            import math
            ppl = math.exp(min(tot / n, 20))
            model.train()
            curve.append((step, round(ppl, 2)))
            best_ppl = min(best_ppl, ppl)
            print(f"  [{mode}] step {step}: PPL={ppl:.2f} ({(time.time()-t0)/60:.1f}min)")
    return {'mode': mode, 'curve': curve, 'best_ppl': round(best_ppl, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='M0', choices=['M0', 'M1', 'M2'])
    ap.add_argument('--steps', type=int, default=5000)
    ns = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = run(ns.mode, ns.steps)
    res['valid'] = res['best_ppl'] <= 25.0
    with open(OUT / f'local_rule_level2_{ns.mode}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"{ns.mode}: best={res['best_ppl']} valid={res['valid']} -> results/mechanism/local_rule_level2_{ns.mode}.json")


if __name__ == '__main__':
    main()
