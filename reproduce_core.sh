#!/bin/bash
# 最小复现：96M 混合 vs TF 单 seed（~1h 本地 RTX 4080）
# 前置：.venv 已创建，WikiText 缓存在 cache/ 目录
PY=./.venv/Scripts/python.exe
COMMON="--d_gate 128 --topk 128 --layers 24 --d_model 512 --ffn_dim 2048
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256
        --max_steps 10000 --warmup_steps 2000 --grad_clip 0.5
        --dataset wikitext --eval_interval 1000 --log_interval 2000
        --amp_dtype fp16 --seed 0"
$PY train.py --model hybrid --no_gating --tf_layers 12 \
    --lr 1e-4 --lr_backbone 5e-4 --exp_name repro_hyb $COMMON
$PY train.py --model transformer --lr 5e-4 \
    --exp_name repro_tf $COMMON
$PY -c "import json; \
h=json.load(open('results/repro_hyb/results.json'))['best_ppl']; \
t=json.load(open('results/repro_tf/results.json'))['best_ppl']; \
print(f'HYB {h:.2f} vs TF {t:.2f} -> 差距 {(1-h/t)*100:+.1f}%')"
