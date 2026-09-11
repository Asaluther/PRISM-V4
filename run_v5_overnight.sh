#!/bin/bash
# ================================================================
# V5 过夜总控：V5.1.2 → 中期汇总 → V5.1.3 → V5.1.4 → 状态文件 → 关机
# 用户授权（2026-09-07）：全部完成后自动关机；任何失败则停止且不关机
# 幂等：所有实验以 results.json 存在为完成标志，可安全重跑
# ================================================================

set -e
set -o pipefail   # 关键：run_wt 的管道中 python 失败必须中断脚本，否则失败后仍会关机
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/v5_overnight.log"
STATUS="results/V5_OVERNIGHT_STATUS.md"

log() { echo "[$(date '+%H:%M:%S')] $@" | tee -a "$LOG"; }

run_wt () {
    NAME=$1; BATCH=$2; ACCUM=$3; STEPS=$4; shift 4
    if [ -f "results/$NAME/results.json" ]; then
        log "skip $NAME"
        return
    fi
    log "START $NAME"
    "$PY" train.py \
        --exp_name "$NAME" --dataset wikitext \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps $STEPS --warmup_steps 500 \
        --lr 3e-4 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        "$@" 2>&1 | grep -E "PPL|完成|Error|OutOfMemory" | tee -a "$LOG"
    log "DONE $NAME"
}

summarize () {
    "$PY" - << 'PYEOF' 2>/dev/null | tee -a "$LOG"
import json, statistics as st
from pathlib import Path
def agg(prefix):
    v = [json.loads(p.read_text())['best_ppl'] for p in Path('results').glob(prefix + '_s*/results.json')]
    v += [json.loads(p.read_text())['best_ppl'] for p in Path('results').glob(prefix + '/results.json')] if Path(f'results/{prefix}/results.json').exists() else []
    return v
for name in ('wt_transformer', 'wt_no_gating', 'wt_twopass_dg64tk64',
             'wt20k_transformer', 'wt20k_no_gating', 'wt20k_twopass_dg64tk64',
             'wt384_transformer', 'wt384_no_gating', 'wt384_twopass_dg64tk64'):
    v = agg(name)
    if v:
        print(f"  {name:<28} n={len(v)} mean={st.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")
PYEOF
}

# ============ V5.1.2 WikiText 41M tokens 主实验 ============
log "===== V5.1.2 WikiText 41M 主实验开始 ====="
for SEED in 0 1 2 3 4; do
    run_wt "wt_transformer_s$SEED"      32 1 5000 --model transformer --seed $SEED
    run_wt "wt_no_gating_s$SEED"        32 1 5000 --model pcn --no_gating --d_gate 64 --topk 64 --seed $SEED
    run_wt "wt_twopass_dg64tk64_s$SEED" 16 2 5000 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --seed $SEED
done
log "===== V5.1.2 完成，中期汇总 ====="
summarize

# ============ V5.1.3 164M tokens 档（20K 步 × 2 seed） ============
log "===== V5.1.3 164M 档开始 ====="
for SEED in 0 1; do
    run_wt "wt20k_transformer_s$SEED"      32 1 20000 --model transformer --seed $SEED
    run_wt "wt20k_no_gating_s$SEED"        32 1 20000 --model pcn --no_gating --d_gate 64 --topk 64 --seed $SEED
    run_wt "wt20k_twopass_dg64tk64_s$SEED" 16 2 20000 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --seed $SEED
done
log "===== V5.1.3 完成，汇总 ====="
summarize

# ============ V5.1.4 d=384 规模组（5000 步，各 1 run） ============
log "===== V5.1.4 d384 组开始 ====="
run_wt "wt384_transformer_s0"      32 1 5000 --model transformer --layers 12 --d_model 384 --ffn_dim 1536 --seed 0
run_wt "wt384_no_gating_s0"        16 2 5000 --model pcn --no_gating --d_gate 64 --topk 64 --layers 12 --d_model 384 --seed 0
run_wt "wt384_twopass_dg64tk64_s0"  8 4 5000 --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --layers 12 --d_model 384 --seed 0
log "===== V5.1.4 完成，汇总 ====="
summarize

# ============ 状态文件 + 关机 ============
{
    echo "# V5 过夜执行状态"
    echo ""
    echo "- 完成时间: $(date)"
    echo "- V5.1.2 / V5.1.3 / V5.1.4 全部执行完毕，汇总数据见 $LOG 与各 results.json"
    echo ""
    echo "## 最终汇总"
    summarize
    echo ""
    echo "系统即将自动关机（用户已授权）。"
} > "$STATUS"

log "全部完成，120 秒后关机（用户已授权）"
cmd //c "shutdown /s /t 120"
