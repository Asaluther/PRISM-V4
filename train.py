#!/usr/bin/env python3
"""PRISM V4 — 统一训练脚本"""

import sys
import os
import json
import time
import argparse
import math
from pathlib import Path

# 项目根目录
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from src.models.pcn import PCNModel
from src.models.transformer import TransformerModel
from src.data.tinystories import get_dataloaders
from src.data.wikitext import get_wikitext_loaders


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    """Wrapper to support both new and old PyTorch API"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, device, model_kwargs=None):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x, **(model_kwargs or {}))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-1
        )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))  # clamp 防溢出
    return avg_loss, ppl


def build_model(args, vocab_size):
    if args.model == 'pcn':
        model = PCNModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.n_heads,
            d_gate=args.d_gate,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            use_bmm_gate=(getattr(args, 'gate_impl', 'bmm') == 'bmm'),
            act_sparse=getattr(args, 'act_sparse', 0.0),
        )
    elif args.model == 'transformer':
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            attn_impl=getattr(args, 'attn_impl', 'sdpa'),
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    no_feedback = getattr(args, 'no_feedback', False)
    no_gating = getattr(args, 'no_gating', False)
    if no_feedback or no_gating:
        print(f"  Ablation: no_feedback={no_feedback}, no_gating={no_gating}")

    return model


def train(args):
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 数据
    print(f"Loading {args.dataset}...")
    if args.dataset == 'wikitext':
        train_loader, val_loader, tokenizer = get_wikitext_loaders(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=2,
        )
    else:
        train_loader, val_loader, tokenizer = get_dataloaders(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=2,
            max_train=50000,
            max_val=5000,
        )
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 模型
    model = build_model(args, vocab_size).to(device)
    n_params = model.count_params()
    print(f"  Model: {args.model}, Params: {n_params / 1e6:.2f}M")

    if device.type == 'cuda':
        print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB (initial)")

    # 优化器（局部规则的参数从 BP 中排除，避免双重更新）
    if args.model == 'pcn' and getattr(args, 'local_rule', 'none') != 'none':
        excl = {'m1': ('W_td', 'W_bu'), 'm2': ('W_td', 'W_bu', 'W_res')}[args.local_rule]
        opt_params = [p for n_, p in model.named_parameters()
                      if not any(n_.endswith(f'{w}.weight') for w in excl)]
        print(f"  局部规则 {args.local_rule}: 排除 {len(list(model.parameters())) - len(opt_params)} 个 BP 参数")
    else:
        opt_params = model.parameters()
    optimizer = AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, args.max_steps)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 输出目录
    exp_name = args.exp_name
    save_dir = Path(args.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_file = save_dir / 'train.log'
    results_data = {
        'config': vars(args),
        'n_params': n_params,
        'vocab_size': vocab_size,
        'val_ppl': [],
    }

    best_ppl = float('inf')
    start_time = time.time()
    step = 0

    # 训练循环
    print(f"\n{'='*70}")
    print(f"Training: {args.model} | {n_params/1e6:.1f}M | {args.max_steps} steps")
    print(f"{'='*70}\n")

    model.train()
    data_iter = iter(train_loader)
    accum = max(1, getattr(args, 'accum_steps', 1))

    # 消融/结构参数在训练与评估间保持一致（首轮实验的教训：透传不一致导致
    # no_gating/no_feedback 的验证走了与训练不同的计算图）
    no_feedback = getattr(args, 'no_feedback', False)
    no_gating = getattr(args, 'no_gating', False)
    model_kwargs = {}
    if args.model == 'pcn':
        model_kwargs = dict(topk=args.topk, no_feedback=no_feedback, no_gating=no_gating,
                            feedback_mode=getattr(args, 'feedback_mode', 'prev_layer'),
                            n_pass=getattr(args, 'n_pass', 1))

    def next_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        return batch[0].to(device), batch[1].to(device)

    while step < args.max_steps:
        optimizer.zero_grad()
        nan_hit = False
        running_loss = 0.0

        with autocast(enabled=(device.type == 'cuda')):
            for _ in range(accum):
                x, y = next_batch()
                logits = model(x, **model_kwargs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-1
                ) / accum
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_hit = True
                    break
                running_loss += loss.item()
                scaler.scale(loss).backward()

        if nan_hit:
            print(f'⚠️ Step {step}: NaN/Inf loss, skipping')
            optimizer.zero_grad()
            step += 1
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # V6 级别2：局部学习规则（在已验证的训练循环内插入，其余流程不变）
        if args.model == 'pcn' and getattr(args, 'local_rule', 'none') != 'none':
            with torch.no_grad():
                Dm = model.d_model
                lr_loc = args.lr_loc if args.lr_loc is not None else args.lr
                for li, blk in enumerate(model.layers):
                    pc = blk.pcn_layer
                    e = getattr(pc, 'last_e', None)
                    if e is None:
                        continue
                    e = e.float()          # AMP 下记录量为 fp16，与 fp32 权重对齐
                    x_l = pc.last_x.float()
                    h_above = pc.last_h_above.float() if pc.last_h_above is not None else None
                    norm = e.size(0) * e.size(1)
                    if args.local_rule in ('m1', 'm2'):
                        if h_above is not None:
                            pc.W_td.weight += lr_loc * (e.reshape(-1, Dm).t() @ h_above.reshape(-1, Dm)) / norm
                        pc.W_bu.weight += lr_loc * (e.reshape(-1, Dm).t() @ x_l.reshape(-1, Dm)) / norm
                    if args.local_rule == 'm2' and li + 1 < len(model.layers):
                        en = getattr(model.layers[li + 1].pcn_layer, 'last_e', None)
                        if en is not None:
                            en = en.float()
                            pc.W_res.weight += lr_loc * (en.reshape(-1, Dm).t() @ x_l.reshape(-1, Dm)) / norm

        step += 1

        # 日志
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (args.max_steps - step) / max(steps_per_sec, 1e-10)
            lr = scheduler.get_last_lr()[0]

            line = (f"Step {step:>6}/{args.max_steps} | "
                    f"loss={running_loss:.4f} | "
                    f"lr={lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"ETA={eta/60:.1f}min")

            if device.type == 'cuda':
                line += f" | VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB"

            print(line)
            with open(log_file, 'a') as f:
                f.write(line + '\n')

        # 验证
        if step % args.eval_interval == 0 or step == args.max_steps:
            eval_loss, ppl = evaluate(model, val_loader, device, model_kwargs)
            elapsed_min = (time.time() - start_time) / 60

            eval_line = (f"  ★ Step {step}: PPL={ppl:.2f} | "
                        f"loss={eval_loss:.4f} | ({elapsed_min:.1f}min)")
            print(eval_line)
            with open(log_file, 'a') as f:
                f.write(eval_line + '\n')

            results_data['val_ppl'].append({
                'step': step,
                'loss': eval_loss,
                'ppl': ppl,
                'time_min': elapsed_min,
            })

            if ppl < best_ppl:
                best_ppl = ppl
                torch.save(model.state_dict(), save_dir / 'best_model.pt')
                print(f"  ★ New best: PPL={best_ppl:.2f}")

            model.train()

    # 最终结果
    total_time = time.time() - start_time
    results_data['best_ppl'] = best_ppl
    results_data['total_time_s'] = total_time
    # 效率记录（证据线 B）：等效 token 预算与吞吐
    tokens_per_step = args.batch_size * accum * args.seq_len
    results_data['budget'] = {
        'optimizer_steps': args.max_steps,
        'accum_steps': accum,
        'effective_batch': args.batch_size * accum,
        'seq_len': args.seq_len,
        'total_tokens_seen': args.max_steps * tokens_per_step,
        'tokens_per_sec': args.max_steps * tokens_per_step / total_time,
    }
    if device.type == 'cuda':
        results_data['budget']['peak_vram_gb'] = torch.cuda.max_memory_allocated() / 1e9

    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"训练完成! Best PPL={best_ppl:.2f} | "
          f"耗时={total_time/60:.1f}min | "
          f"Params={n_params/1e6:.2f}M")
    print(f"结果: {save_dir / 'results.json'}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='PRISM V4 Training')
    parser.add_argument('--model', type=str, default='pcn', choices=['pcn', 'transformer'])
    parser.add_argument('--dataset', type=str, default='tinystories',
                        choices=['tinystories', 'wikitext'])
    parser.add_argument('--attn_impl', type=str, default='sdpa', choices=['sdpa', 'mha', 'decay'],
                        help='Transformer 注意力实现：sdpa=标准快路径（默认）；mha=兼容旧 checkpoint')
    parser.add_argument('--no_feedback', action='store_true', help='PCN: 去掉 top-down 预测')
    parser.add_argument('--no_gating', action='store_true', help='PCN: 用标准 attention 替代误差门控')
    parser.add_argument('--feedback_mode', type=str, default='prev_layer',
                        choices=['prev_layer', 'two_pass'],
                        help='prev_layer: 首轮实现（浅层预测深层，非真 top-down）；'
                             'two_pass: 真 top-down（pass2 用更深一层做预测来源）')
    parser.add_argument('--local_rule', type=str, default='none',
                        choices=['none', 'm1', 'm2'],
                        help='V6 级别2 局部学习规则：m1=W_td/W_bu 局部(其余BP)；'
                             'm2=主路径全局部(W_up/W_res 用上一步下层误差)')
    parser.add_argument('--lr_loc', type=float, default=None,
                        help='局部规则学习率（默认=args.lr；P1a 对称扫描用）')
    parser.add_argument('--n_pass', type=int, default=1,
                        help='K-pass 迭代推断（P1b：K>1 时末 pass 误差用于局部规则）')
    parser.add_argument('--act_sparse', type=float, default=0.0,
                        help='V7 激活稀疏化比例（0=不稀疏，0.25=保留 25%% 最大激活）')
    parser.add_argument('--gate_impl', type=str, default='bmm', choices=['bmm', 'loop'],
                        help='门控实现：bmm=等价加速形式（默认）；loop=原 chunk 循环（对照）')
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--ffn_dim', type=int, default=1024)
    parser.add_argument('--d_gate', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--topk', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='梯度累积微批次数：等效 batch = batch_size × accum_steps')
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
