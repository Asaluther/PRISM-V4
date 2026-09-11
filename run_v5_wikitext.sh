#!/bin/bash
# ================================================================
# V5.1.2 — WikiText-103 主实验（真实域试金石）
# 三配置 × 5 seed × 41M tokens（5000 步 × batch 32 × seq 256，与 V4 同预算）
#
#   wt_transformer       标准对照
#   wt_no_gating         误差流 + attention 聚合（V4 最稳冠军 1.123±0.012）
#   wt_twopass           真 top-down + 门控聚合（d_gate64/tk64 最优带宽）
#
# Go/No-Go：PCN 最强变体同预算 PPL 不劣于 Transformer → V6 启动
# 依赖：cache/wikitext_*_int32.npy 已构建（run 一次 src/data/wikitext.py）
# ================================================================

set -e
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v5_wikitext_pipeline.log"

run_exp () {
    NAME=$1; shift
    BATCH=$2; ACCUM=$3; shift 3
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date) =====" | tee -a "$LOG"
    # two_pass + d_gate64 在 batch 32 下显存打满（实测 allocator 减速 10 倍），
    # 统一 batch 16 × accum 2 = 等效 batch 32，token 预算不变
    "$PY" train.py \
        --exp_name "$NAME" \
        --dataset wikitext \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 500 \
        "$@" 2>&1 | grep -E "PPL|完成|Error" | tee -a "$LOG"
    echo "===== $NAME done $(date) =====" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
    run_exp "wt_transformer_s$SEED"       32 1 --model transformer --seed $SEED
    run_exp "wt_no_gating_s$SEED"         32 1 --model pcn --no_gating --d_gate 64 --topk 64 --seed $SEED
    run_exp "wt_twopass_dg64tk64_s$SEED"  16 2 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --seed $SEED
done

echo "V5.1.2 ALL DONE $(date)" | tee -a "$LOG"
