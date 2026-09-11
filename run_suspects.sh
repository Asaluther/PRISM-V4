#!/bin/bash
# ================================================================
# 置信度疑点验证管线（WikiText Go 判定后的两个剩余疑点）
# 疑点1 基线强度：SDPA vs MHA 等价对照 + 基线 lr 扫描
# 疑点2 欠训练效应：20K 步（164M tokens）三配置走势
# 判读：
#   S1a: SDPA PPL ≈ 35.98 → 基线稳健；显著更低 → 以 SDPA 为新基线重判
#   S1b: 最优 lr 下基线仍 >>16.5 → 排除「基线被压制」
#   S2:  5K→20K 差距保持/扩大 → 真优势；显著收敛 → 欠训练伪影警报
# ================================================================

set -e
set -o pipefail
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/suspects_pipeline.log"

run_exp () {
    NAME=$1; BATCH=$2; ACCUM=$3; STEPS=$4; shift 4
    if [ -f "results/$NAME/results.json" ]; then
        echo "skip $NAME" | tee -a "$LOG"
        return
    fi
    echo "===== $NAME start $(date +%H:%M) =====" | tee -a "$LOG"
    "$PY" train.py \
        --exp_name "$NAME" \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps $STEPS --warmup_steps 500 \
        --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        "$@" 2>&1 | grep -E "PPL|完成|Error|OutOfMemory" | tee -a "$LOG"
    echo "===== $NAME done $(date +%H:%M) =====" | tee -a "$LOG"
}

# ---- 疑点 1a：SDPA 基线对照（TinyStories 5K，对照 MHA 版 35.98）----
run_exp "ts_sdpa_tf_s0" 32 1 5000 --model transformer --attn_impl sdpa --lr 3e-4 --seed 0

# ---- 疑点 1b：SDPA 基线 lr 扫描 ----
run_exp "ts_sdpa_tf_lr1e4_s0" 32 1 5000 --model transformer --attn_impl sdpa --lr 1e-4 --seed 0
run_exp "ts_sdpa_tf_lr1e3_s0" 32 1 5000 --model transformer --attn_impl sdpa --lr 1e-3 --seed 0

# ---- 疑点 2：WikiText 20K 步（164M tokens）----
run_exp "wt20k_causal_tf_s0"     32 1 20000 --dataset wikitext --model transformer --attn_impl sdpa --lr 3e-4 --seed 0
run_exp "wt20k_causal_nogate_s0" 32 1 20000 --dataset wikitext --model pcn --no_gating --d_gate 64 --topk 64 --lr 3e-4 --seed 0
run_exp "wt20k_causal_2p_s0"     16 2 20000 --dataset wikitext --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --lr 3e-4 --seed 0

echo "SUSPECTS ALL DONE $(date +%H:%M)" | tee -a "$LOG"
