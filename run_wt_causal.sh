#!/bin/bash
# ================================================================
# 因果版 WikiText 重做（V5.1.2R）—— 真实域试金石
# 三配置 × 5 seed × 41M tokens（5000 步 × 等效 batch 32 × seq 256）
# 依赖：cache/wikitext_*_int32.npy 已就绪
# Go/No-Go：PCN 最强变体同预算 PPL 不劣于 Transformer → V6
# ================================================================

set -e
set -o pipefail
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/wt_causal_pipeline.log"

run_exp () {
    NAME=$1; BATCH=$2; ACCUM=$3; shift 3
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date +%H:%M) =====" | tee -a "$LOG"
    "$PY" train.py \
        --exp_name "$NAME" --dataset wikitext \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        "$@" 2>&1 | grep -E "PPL|完成|Error|OutOfMemory" | tee -a "$LOG"
    echo "===== $NAME done $(date +%H:%M) =====" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
    run_exp "wt_causal_transformer_s$SEED" 32 1 --model transformer --seed $SEED
    run_exp "wt_causal_no_gating_s$SEED"   32 1 --model pcn --no_gating --d_gate 64 --topk 64 --seed $SEED
    run_exp "wt_causal_twopass_s$SEED"     16 2 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --seed $SEED
done

echo "WT CAUSAL ALL DONE $(date +%H:%M)" | tee -a "$LOG"
