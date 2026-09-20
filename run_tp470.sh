#!/usr/bin/env bash
# B4 规模/数据比解耦实验——96M 模型 @ ~470 tok/param（~1.7h 串行，6 runs）
#
# 问题：HYB 优势收窄（96M@853t/p: 36.4% → 330M@470t/p: 11.2% 调优口径）
# 同时移动了规模与数据比两个变量。本队列只动数据比（规模固定 96M 档）：
# 5700 步 × 8192 tok/步 = 46.7M tokens → HYB 96.0M: 486 t/p / TF 101.5M: 460 t/p
# （镜像 330M 点的 496/463）。
#
# 预注册判读（对照 330M@470 的逐 seed 配对 5.3/6.5/35.5）：
#   96M@470 差距落 5~15% → 数据比驱动（倒 U 左坡良性收窄）
#   仍 ≈36%            → 规模驱动（架构天花板叙事）
#
# 其余配置与 96M 基线（wt_094m_hyb_s0 / wt_200m_tf_lr5e-4 系）逐字段一致，
# warmup 1140 = 20% 步数惯例。
set -u
PY=./.venv/Scripts/python.exe
COMMON="--d_gate 128 --topk 128 --layers 24 --d_model 512 --n_heads 4 --ffn_dim 2048
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256
        --max_steps 5700 --warmup_steps 1140 --grad_clip 0.5 --dataset wikitext
        --eval_interval 1000 --log_interval 2000 --amp_dtype fp16"

for S in 0 1 2; do
  $PY train.py --model hybrid --no_gating --tf_layers 12 \
      --lr 1e-4 --lr_backbone 5e-4 --seed $S \
      --exp_name wt_096m_hyb_tp470_s$S $COMMON
  $PY train.py --model transformer --lr 5e-4 --seed $S \
      --exp_name wt_101m_tf_tp470_s$S $COMMON
done
