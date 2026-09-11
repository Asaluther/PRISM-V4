#!/bin/bash
# ================================================================
# V4.x#4 + #5 — 反馈 two_pass 判定 + 门控超参敏感性扫描
# 在 run_v4x_seeds.sh（#1）完成后运行
#
# #4 三臂判定数据来源：
#   prev_layer（伪反馈） = pcn_v2_fixed        PPL 14.05（已有）
#   裁撤                 = pcn_v2_no_feedback  PPL 2.07（已有）
#   two_pass（真 top-down）= 本脚本运行
# 判定：two_pass 显著优于裁撤（>10%）→ 保留反馈改实现；
#       否则 → 裁撤，PCN 重定义为误差门控前馈网络
# ================================================================

set -e
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v4x_followup_pipeline.log"

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
        "$@" 2>&1 | grep -E "PPL|完成|Error" | tee -a "$LOG"
    echo "===== $NAME done $(date) =====" | tee -a "$LOG"
}

# ---- V4.x#4: 真 top-down 反馈（two_pass）----
run_exp pcn_v2_twopass --model pcn --feedback_mode two_pass --d_gate 32 --topk 32 --seed 0

# ---- V4.x#5: d_gate × topk 敏感性（no_feedback 版，与 #1 的 s1 seed 对齐）----
for DG in 16 32 64; do
    for TK in 16 32 64; do
        run_exp "scan_nf_dg${DG}_tk${TK}" --model pcn --no_feedback \
            --d_gate $DG --topk $TK --seed 1
    done
done

echo "FOLLOWUP ALL DONE $(date)" | tee -a "$LOG"
