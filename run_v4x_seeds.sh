#!/bin/bash
# ================================================================
# V4.x#1 — 多 seed 复验
# 3 配置 × seed 1..5：确认 PPL 排序（no_gating / no_feedback / transformer）
# 与 run_v4_fixed.sh 完全同预算协议。
# ================================================================

set -e
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v4x_seeds_pipeline.log"

run_exp () {
    NAME=$1; shift
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date) =====" | tee -a "$LOG"
    "$PY" train.py \
        --exp_name "$NAME" \
        --batch_size 32 --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        --d_gate 32 --topk 32 \
        "$@" 2>&1 | grep -E "Step|PPL|完成|Error" | tee -a "$LOG"
    echo "===== $NAME done $(date) =====" | tee -a "$LOG"
}

for SEED in 1 2 3 4 5; do
    run_exp "pcn_v2_no_feedback_s$SEED" --model pcn --no_feedback --seed $SEED
    run_exp "pcn_v2_no_gating_s$SEED"   --model pcn --no_gating   --seed $SEED
    run_exp "transformer_s$SEED"        --model transformer       --seed $SEED
done

echo "ALL SEED RUNS DONE $(date)" | tee -a "$LOG"
