#!/bin/bash
# 夜间 D 队列（2026-09-11）：数据量切片 + 曲线加密 + 长训练稳定性 + 块二分补强
cd "$(dirname "$0")"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=".venv/Scripts/python.exe"
LOG="results/overnight_d.log"
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $@" | tee -a "$LOG"; }

# 步数 = tokens / (batch32 * seq256) = tokens/8192
run_d() {  # NAME STEPS MODEL_ARGS...
    NAME=$1; STEPS=$2; shift 2
    [ -f "results/$NAME/results.json" ] && { log "skip $NAME"; return 0; }
    log "START $NAME ($STEPS steps)"
    "$PY" train.py --exp_name "$NAME" --batch_size 32 --seq_len 256 --max_seq_len 256 \
        --max_steps $STEPS --warmup_steps 200 --weight_decay 0.1 \
        --eval_interval 300 --log_interval 1000 --grad_clip 0.5 \
        "$@" >> "$LOG" 2>&1
    [ -f "results/$NAME/results.json" ] && log "DONE $NAME" || log "FAIL $NAME (continue)"
}

gpu_line() {
    log "===== D1: 数据量切片（d256 TS，tokens 2.5M-20M；41M 已有）====="
    for SEED in 0 1; do
        for M in 2.5 5 10 20; do
            STEPS=$(python -c "print(int($M*1000000/8192))")
            run_d "dslice_tf_${M}M_s$SEED" $STEPS --model transformer --attn_impl sdpa --lr 1e-3 --seed $SEED
            run_d "dslice_ng_${M}M_s$SEED" $STEPS --model pcn --no_gating --d_gate 64 --topk 64 --lr 5e-4 --seed $SEED
        done
    done
    log "===== D2: WT d512 补 seed（n=2 -> 5）====="
    for SEED in 2 3 4; do
        [ -f "results/w512_tf_s$SEED/results.json" ] && { log "skip w512_tf_s$SEED"; } || {
            log "START w512_tf_s$SEED"
            "$PY" train.py --exp_name "w512_tf_s$SEED" --dataset wikitext --model transformer \
                --attn_impl sdpa --lr 1e-3 --layers 12 --d_model 512 --ffn_dim 2048 \
                --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 \
                --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $SEED >> "$LOG" 2>&1
            log "DONE/checked w512_tf_s$SEED"
        }
        [ -f "results/w512_ng_s$SEED/results.json" ] && { log "skip w512_ng_s$SEED"; } || {
            log "START w512_ng_s$SEED"
            "$PY" train.py --exp_name "w512_ng_s$SEED" --dataset wikitext --model pcn --no_gating \
                --d_gate 64 --topk 64 --lr 5e-4 --layers 12 --d_model 512 --ffn_dim 2048 \
                --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 \
                --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $SEED >> "$LOG" 2>&1
            log "DONE/checked w512_ng_s$SEED"
        }
    done
    log "===== D3: NG 长训练稳定性对照（WT 20K，lr 3e-4 vs 2e-4）====="
    for LR in 3e-4 2e-4; do
        NAME="wt20k_ng_lr${LR}_s0"
        [ -f "results/$NAME/results.json" ] && { log "skip $NAME"; continue; }
        log "START $NAME"
        "$PY" train.py --exp_name "$NAME" --dataset wikitext --model pcn --no_gating \
            --d_gate 64 --topk 64 --lr $LR --grad_clip 0.5 \
            --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 20000 --warmup_steps 500 \
            --weight_decay 0.1 --eval_interval 2000 --log_interval 2000 --seed 0 >> "$LOG" 2>&1
        log "DONE/checked $NAME"
    done
    log "===== GPU line end ====="
}

cpu_line() {
    log "===== CPU: block_ablation 补 seed（3->5）====="
    # gate_source_v2 的校准+实验格补 2 seed
    for S in 3 4; do
        log "START copy_mechanism seed $S (补充)"
        CUDA_VISIBLE_DEVICES=-1 "$PY" copy_mechanism.py --seed $S \
            > "results/mechanism/copy_mechanism_s${S}.out" 2>&1 \
            && log "DONE cm seed $S" || log "FAIL cm seed $S (continue)"
    done
    log "===== CPU line end ====="
}

gpu_line &
GPU_PID=$!
cpu_line &
CPU_PID=$!
log "GPU PID=$GPU_PID CPU PID=$CPU_PID"
wait $GPU_PID $CPU_PID
log "===== lines joined, writing report ====="

"$PY" - << 'PYEOF' > results/OVERNIGHT_D_STATUS.md 2>>results/overnight_d.log
import json
from pathlib import Path
from statistics import mean, stdev
print("# 夜间 D 队列执行报告\n")
print(f"- 完成时间: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}\n")

print("## D1 数据量切片（d256 TS；41M 点 = ts_causal 5-seed 已有）\n")
print("| tokens | TF (n=2) | NG (n=2) | NG 优势 |")
print("|---|---|---|---|")
for m in ('2.5', '5', '10', '20'):
    tf = [json.loads(p.read_text())['best_ppl'] for p in Path('results').glob(f'dslice_tf_{m}M_s*/results.json')]
    ng = [json.loads(p.read_text())['best_ppl'] for p in Path('results').glob(f'dslice_ng_{m}M_s*/results.json')]
    if tf and ng:
        adv = (1 - mean(ng)/mean(tf)) * 100
        print(f"| {m}M | {mean(tf):.2f} | {mean(ng):.2f} | {adv:+.1f}% |")

print("\n## D2 WT d512（补 seed 后）\n")
for name in ('w512_tf', 'w512_ng'):
    v = [json.loads(p.read_text())['best_ppl'] for p in sorted(Path('results').glob(f'{name}_s*/results.json'))]
    if v:
        print(f"- {name}: n={len(v)} mean={mean(v):.2f} ± {stdev(v) if len(v)>1 else 0:.2f}")

print("\n## D3 NG 长训练稳定性（WT 20K；对照 TF@1e-3=53.1 / NG@5e-4 曾崩到 285）\n")
for lr in ('3e-4', '2e-4'):
    p = Path(f'results/wt20k_ng_lr{lr}_s0/results.json')
    if p.exists():
        r = json.loads(p.read_text())
        print(f"- NG@{lr} 20K: {r['best_ppl']:.1f}")

print("\n## 块二分补 seed（copy_mechanism s3/s4）\n")
import os
for s in (3, 4):
    p = Path(f'results/mechanism/copy_mechanism_s{s}.json')
    print(f"- seed{s}: {'✓ 产出' if p.exists() else '未产出'}")

print("\n---\n完整日志: results/overnight_d.log | 系统即将关机（用户已授权）")
PYEOF

log "报告已生成 results/OVERNIGHT_D_STATUS.md"
log "全部完成，120 秒后关机"
cmd //c "shutdown /s /t 120"
