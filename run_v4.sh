#!/bin/bash
# ================================================================
# PRISM V4 训练管线
# 实验序列:
#   1. Transformer baseline (TinyStories, 5K steps)
#   2. PCN model (TinyStories, 5K steps)
#   3. PCN ablations (no_feedback, no_gating)
#
# 资源约束:
#   - System RAM: tokenization happens in-memory, use small max_train
#   - GPU VRAM: PCN N² gate computation, use batch_size=4 + seq_len=128 for PCN
#   - Network: offline mode (HF_DATASETS_OFFLINE=1)
# ================================================================

set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

V4_ROOT="/media/asaluther/3C1C221E1C21D424/PRISM V4"
RESULTS="$V4_ROOT/results"
LOG="$RESULTS/v4_pipeline.log"

mkdir -p "$RESULTS"

log() { echo "$@" | tee -a "$LOG"; }

log ""
log "============================================"
log "PRISM V4 Pipeline 启动: $(date)"
log "============================================"

cd "$V4_ROOT"

# ================================================================
# 实验 1: Transformer baseline
# ================================================================
EXP="transformer_baseline"
if [ -f "$RESULTS/$EXP/results.json" ]; then
    log "⏭️  $EXP 已完成，跳过"
else
    log ""
    log "========================================"
    log "[1/4] Transformer baseline 开始"
    log "  $(date)"
    log "========================================"
    python3 train.py \
        --model transformer \
        --max_steps 5000 \
        --eval_interval 1000 \
        --log_interval 200 \
        --batch_size 32 \
        --seq_len 256 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --warmup_steps 500 \
        --exp_name "$EXP" \
        2>&1 | tee -a "$LOG"
    log "[1/4] Transformer baseline 完成: $(date)"
fi

# ================================================================
# 实验 2: PCN v1
# ================================================================
EXP="pcn_v1"
if [ -f "$RESULTS/$EXP/results.json" ]; then
    log "⏭️  $EXP 已完成，跳过"
else
    log ""
    log "========================================"
    log "[2/4] PCN v1 开始"
    log "  $(date)"
    log "========================================"
    python3 train.py \
        --model pcn \
        --max_steps 5000 \
        --eval_interval 1000 \
        --log_interval 200 \
        --batch_size 4 \
        --seq_len 128 \
        --max_seq_len 128 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --warmup_steps 500 \
        --d_gate 32 \
        --topk 32 \
        --exp_name "$EXP" \
        2>&1 | tee -a "$LOG"
    log "[2/4] PCN v1 完成: $(date)"
fi

# ================================================================
# 实验 3: PCN ablation — no_feedback
# ================================================================
EXP="pcn_no_feedback"
if [ -f "$RESULTS/$EXP/results.json" ]; then
    log "⏭️  $EXP 已完成，跳过"
else
    log ""
    log "========================================"
    log "[3/4] PCN no_feedback 消融开始"
    log "  $(date)"
    log "========================================"
    python3 train.py \
        --model pcn \
        --no_feedback \
        --max_steps 5000 \
        --eval_interval 1000 \
        --log_interval 200 \
        --batch_size 4 \
        --seq_len 128 \
        --max_seq_len 128 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --warmup_steps 500 \
        --d_gate 32 \
        --topk 32 \
        --exp_name "$EXP" \
        2>&1 | tee -a "$LOG"
    log "[3/4] PCN no_feedback 完成: $(date)"
fi

# ================================================================
# 实验 4: PCN ablation — no_gating (standard attention)
# ================================================================
EXP="pcn_no_gating"
if [ -f "$RESULTS/$EXP/results.json" ]; then
    log "⏭️  $EXP 已完成，跳过"
else
    log ""
    log "========================================"
    log "[4/4] PCN no_gating 消融开始"
    log "  $(date)"
    log "========================================"
    python3 train.py \
        --model pcn \
        --no_gating \
        --max_steps 5000 \
        --eval_interval 1000 \
        --log_interval 200 \
        --batch_size 4 \
        --seq_len 128 \
        --max_seq_len 128 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --warmup_steps 500 \
        --d_gate 32 \
        --topk 32 \
        --exp_name "$EXP" \
        2>&1 | tee -a "$LOG"
    log "[4/4] PCN no_gating 完成: $(date)"
fi

# ================================================================
# 管线完成
# ================================================================
log ""
log "============================================"
log "PRISM V4 管线全部完成: $(date)"
log "============================================"
log ""
log "结果目录:"
for d in "$RESULTS"/*/; do
    if [ -f "$d/results.json" ]; then
        log "  ✅ $(basename "$d")"
    fi
done
