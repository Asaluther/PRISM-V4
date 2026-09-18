#!/usr/bin/env bash
# 200-300M 规模外推（W1）— d1024/L24, 20K 步, seed 0, 本地 RTX 4080
#
# 规格：V5 报告 Q5 预注册最小方案（d1024/24L 单点 × 20K 步）。
# 修正旧判断：Q5 曾评「4080 约不可行」系纸面估算；实测冒烟（HYB 330M）
#   VRAM 6.01GB / 2.1 steps/s → 单模型 20K 步 ≈ 2.7h，本地可行。
#
# 队列（~10.7h 串行）：
#   1. HYB 12+12  骨干5e-4/头1e-4   （101M 调优值直承）
#   2. TF  24L    lr 5e-4           （101M 调优值直承）
#   3. TF  24L    lr 2.5e-4         （夹逼点——防 lr 随规模漂移伪影，v5 偏差#6 教训）
#   4. PCN 24L    lr 1e-4           （101M 调优值直承，稳定性天花板参照）
#
# 诚实边界：单 seed；lr 未重扫（TF 有双点夹逼，HYB/PCN 单点直承——若轨迹
# 异常需补 2.5e-4 臂）；d_gate/topk 维持 128（no_gating 下近残留参量）。
set -u
PY=./.venv/Scripts/python.exe
COMMON="--layers 24 --d_model 1024 --n_heads 8 --ffn_dim 4096 --d_gate 128 --topk 128
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256
        --max_steps 20000 --warmup_steps 4000 --grad_clip 0.5 --dataset wikitext
        --eval_interval 1000 --log_interval 2000 --seed 0 --amp_dtype fp16"

$PY train.py --model hybrid --tf_layers 12 --lr 1e-4 --lr_backbone 5e-4 \
    --exp_name wt_330m_hyb_lr5e-4_s0 $COMMON
$PY train.py --model transformer --lr 5e-4  \
    --exp_name wt_353m_tf_lr5e-4_s0 $COMMON
$PY train.py --model transformer --lr 2.5e-4 \
    --exp_name wt_353m_tf_lr2.5e-4_s0 $COMMON
$PY train.py --model pcn --no_gating --lr 1e-4 \
    --exp_name wt_306m_pcn_lr1e-4_s0 $COMMON
