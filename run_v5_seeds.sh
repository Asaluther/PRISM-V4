#!/bin/bash
# ================================================================
# V5.0.1 — two_pass + 最优带宽（d_gate64/tk64）× 5 seed
# 判定纯 PCN 路线的真实水平与稳定性：
#   均值 1.1-1.5 且 5/5 相变 → 纯 PCN 主线成立
#   仍 bimodal → 主线转 no_gating，two_pass 降级为组件
# two_pass 计算量 ×2，每 run 约 35 分钟，总 ~3h
# ================================================================

set -e
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v5_pipeline.log"

run_exp () {
    NAME=$1; shift
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date) =====" | tee -a "$LOG"
    # batch 16 × accum 2 = 等效 batch 32（预算不变）。
    # 实测 batch 32 + d_gate64 + two_pass 显存 15.9GB 打满 → allocator 反复整理，
    # 速度掉到 <0.73 steps/s；batch 16 解除显存压力。
    "$PY" train.py \
        --exp_name "$NAME" \
        --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 500 \
        --feedback_mode two_pass --d_gate 64 --topk 64 \
        "$@" 2>&1 | grep -E "PPL|完成|Error" | tee -a "$LOG"
    echo "===== $NAME done $(date) =====" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
    run_exp "v5_twopass_dg64tk64_s$SEED" --model pcn --seed $SEED
done

echo "V5.0.1 ALL DONE $(date)" | tee -a "$LOG"
