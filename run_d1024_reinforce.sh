#!/usr/bin/env bash
# 330M 补强队列（~12.5h 串行）——修复昨晚 HYB 漏传 --no_gating 的配置失误
#
# 背景：run_d1024_extrap.sh 的 HYB 行漏传 --no_gating（results.json
#   no_gating:false 实证），与 96M 全系混合模型（no_gating:true）跨配置。
#   本队列主线全部按协议重训；昨晚 gating 点（wt_330m_hyb_lr5e-4_s0，
#   PPL 36.50）保留为单点对照，适应侧对照见 adapt330.py 的 hyb_gate_s0。
#
# 队列：
#   1-3. HYB ng 主线 s0/s1/s2（骨干 5e-4 / 头 1e-4，101min/run）
#   4-5. HYB ng 骨干 lr 夹逼 2.5e-4 / 1e-3（s0）
#   6-7. TF s1/s2 @ 5e-4（s0=41.72 沿用；123min/run）
# PCN 不重播：已软发散的参照点，单 seed 足够（报告注明）。
set -u
PY=./.venv/Scripts/python.exe
COMMON="--d_gate 128 --topk 128 --layers 24 --d_model 1024 --n_heads 8 --ffn_dim 4096
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256
        --max_steps 20000 --warmup_steps 4000 --grad_clip 0.5 --dataset wikitext
        --eval_interval 1000 --log_interval 2000 --amp_dtype fp16"

for S in 0 1 2; do
  $PY train.py --model hybrid --no_gating --tf_layers 12 \
      --lr 1e-4 --lr_backbone 5e-4 --seed $S \
      --exp_name wt_330m_hyb_ng_lr5e-4_s$S $COMMON
done
$PY train.py --model hybrid --no_gating --tf_layers 12 \
    --lr 1e-4 --lr_backbone 2.5e-4 --seed 0 \
    --exp_name wt_330m_hyb_ng_lr2.5e-4_s0 $COMMON
$PY train.py --model hybrid --no_gating --tf_layers 12 \
    --lr 1e-4 --lr_backbone 1e-3 --seed 0 \
    --exp_name wt_330m_hyb_ng_lr1e-3_s0 $COMMON
$PY train.py --model transformer --lr 5e-4 --seed 1 \
    --exp_name wt_353m_tf_lr5e-4_s1 $COMMON
$PY train.py --model transformer --lr 5e-4 --seed 2 \
    --exp_name wt_353m_tf_lr5e-4_s2 $COMMON
