#!/bin/bash
# ================================================================
# 因果版多 seed 复验（TinyStories，3 配置 × 5 seed）
# s0 已完成（transformer 35.98 / no_gating 16.56 / twopass 19.87）
# 目标：确认 54%/45% 优势的 seed 稳定性
# ================================================================

set -e
set -o pipefail
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/ts_causal_seeds.log"

run_exp () {
    NAME=$1; BATCH=$2; ACCUM=$3; shift 3
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date +%H:%M) =====" | tee -a "$LOG"
    "$PY" train.py \
        --exp_name "$NAME" \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        "$@" 2>&1 | grep -E "PPL|完成|Error|OutOfMemory" | tee -a "$LOG"
    echo "===== $NAME done $(date +%H:%M) =====" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
    run_exp "ts_causal_transformer_s$SEED" 32 1 --model transformer --seed $SEED
    run_exp "ts_causal_no_gating_s$SEED"   32 1 --model pcn --no_gating --d_gate 64 --topk 64 --seed $SEED
    run_exp "ts_causal_twopass_s$SEED"     16 2 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --seed $SEED
done

echo "CAUSAL SEEDS ALL DONE $(date +%H:%M)" | tee -a "$LOG"
