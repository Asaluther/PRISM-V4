#!/usr/bin/env python3
"""c3_seq128 — seq 长度敏感性探针（评审 Q5 的最廉价检查）

问题（Q5）：tokens/param 峰位与倒 U 规律是否对 seq 长度敏感？
设计：96M 档 HYB + TF @ seq128（对照基线 seq256），10K 步 × 等效 batch 32
= 82M tokens（与 96M 基线完全同预算）→ 对比三方 PPL 序。

判读：
  - HYB 优势保持（两架构同向变化）→ 规律对 seq 不敏感
  - 优势移动 → 峰位的 seq 依赖性需在论文中标注

用法：.venv/Scripts/python.exe c3_seq128.py   (本地 GPU)
输出：results/wt_096m_hyb_seq128_s0/ + results/wt_101m_tf_seq128_s0/
"""
import subprocess, sys, shlex
from pathlib import Path

VENV_PY = str(Path(__file__).parent / '.venv' / 'Scripts' / 'python.exe')
COMMON = ("--d_gate 128 --topk 128 --layers 24 --d_model 512 --n_heads 4 "
          "--ffn_dim 2048 --batch_size 16 --accum_steps 2 --max_seq_len 128 "
          "--seq_len 128 --max_steps 10000 --warmup_steps 2000 --grad_clip 0.5 "
          "--dataset wikitext --eval_interval 1000 --log_interval 2000 "
          "--amp_dtype fp16 --seed 0")

cmds = [
    [VENV_PY, 'train.py', '--model', 'hybrid', '--no_gating', '--tf_layers', '12',
     '--lr', '1e-4', '--lr_backbone', '5e-4', '--exp_name', 'wt_096m_hyb_seq128_s0']
     + shlex.split(COMMON),
    [VENV_PY, 'train.py', '--model', 'transformer',
     '--lr', '5e-4', '--exp_name', 'wt_101m_tf_seq128_s0'] + shlex.split(COMMON),
]

for c in cmds:
    print(f'\n>>> launching: {c[1][:20]}... {c[-1]}')
    r = subprocess.run(c)
    if r.returncode != 0:
        print(f'!!! 失败 (exit {r.returncode})')
        sys.exit(1)
print('\nC3s 全部完成')
