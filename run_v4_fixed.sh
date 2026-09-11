#!/bin/bash
# ================================================================
# PRISM V4 — 修复后重跑管线（2026-09-07）
#
# 修复内容（依据 results/diagnosis/ 诊断证据）：
#   1. train.py evaluate 透传消融参数 —— 首轮 no_gating/no_feedback
#      的验证走了与训练不同的计算图，结果无效
#   2. 训练预算对齐 transformer_baseline：batch 32 × seq 256 × 5000 步
#      （首轮 PCN batch 4 × seq 128，token 预算仅为对照的 1/16）
#   3. pcn.py init_mode='fixed'：W_res 恒等初始化（修复 12 层梯度衰减 316×）
#      + 门控输入 LayerNorm 与 Xavier 初始化（修复门控死锁 g≡0.5）
#   4. no_gating 消融语义对齐 V4_PLAN 实验 2：保留误差流与反馈，
#      仅将误差门控聚合换成标准 attention（首轮实现是纯 attention 堆叠）
#
# 环境注意：HF 离线变量必须在 shell 层 export（tinystories.py 的
# setdefault 时机对 datasets 5.x 不完全生效，实证见 2026-09-07 干跑）
# ================================================================

set -e
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v4_fixed_pipeline.log"

run_exp () {
    NAME=$1; shift
    if [ -f "results/$NAME/results.json" ]; then
        echo "⏭️  $NAME 已完成，跳过" | tee -a "$LOG"
        return
    fi
    echo "" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    echo "[RUN] $NAME 开始: $(date)" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    "$PY" train.py \
        --exp_name "$NAME" \
        --batch_size 32 --seq_len 256 --max_seq_len 256 \
        --max_steps 5000 --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 100 \
        --d_gate 32 --topk 32 \
        "$@" 2>&1 | tee -a "$LOG"
    echo "[RUN] $NAME 完成: $(date)" | tee -a "$LOG"
}

# 与 transformer_baseline（batch 32 / seq 256 / 5000 步）完全同预算
run_exp pcn_v2_fixed       --model pcn
run_exp pcn_v2_no_feedback --model pcn --no_feedback
run_exp pcn_v2_no_gating   --model pcn --no_gating

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "PRISM V4 修复版管线全部完成: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
