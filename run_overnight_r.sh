#!/bin/bash
# ================================================================
# 夜间 R 队列总控（2026-09-09）
# 用户指令：6 项队列 -> 完成后记录（无论好坏/异常）-> 关机
# 设计：不用 set -e（单项失败继续）；GPU 串行线 & CPU 并行线；
#       wait 汇合 -> 汇总报告 -> shutdown /t 120
# ================================================================

cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PY=".venv/Scripts/python.exe"
LOG="results/overnight_r.log"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $@" | tee -a "$LOG"; }

run_wt() {
    NAME=$1; BATCH=$2; ACCUM=$3; STEPS=$4; DS=$5; shift 5
    if [ -f "results/$NAME/results.json" ]; then
        log "skip $NAME"
        return 0
    fi
    log "START $NAME"
    "$PY" train.py --exp_name "$NAME" --dataset "$DS" \
        --batch_size $BATCH --accum_steps $ACCUM --seq_len 256 --max_seq_len 256 \
        --max_steps $STEPS --warmup_steps 500 --weight_decay 0.1 \
        --eval_interval 1000 --log_interval 1000 \
        "$@" >> "$LOG" 2>&1
    if [ -f "results/$NAME/results.json" ]; then log "DONE $NAME"; else log "FAIL $NAME (continue)"; fi
}

gpu_line() {
    log "===== GPU line start ====="
    run_wt "o384_tf" 32 1 5000 tinystories --model transformer --attn_impl sdpa --lr 1e-3 --layers 12 --d_model 384 --ffn_dim 1536 --seed 0
    run_wt "o384_ng" 32 1 5000 tinystories --model pcn --no_gating --d_gate 64 --topk 64 --lr 3e-4 --layers 12 --d_model 384 --seed 0
    run_wt "o384_2p" 8 4 5000 tinystories --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --lr 3e-4 --layers 12 --d_model 384 --seed 0
    run_wt "wt20k_tf_lr1e3_s0" 32 1 20000 wikitext --model transformer --attn_impl sdpa --lr 1e-3 --seed 0
    run_wt "wt_ng_lr5e4_s0"  32 1 5000 wikitext --model pcn --no_gating --d_gate 64 --topk 64 --lr 5e-4 --seed 0
    run_wt "wt_2p_lr5e4_s0" 16 2 5000 wikitext --model pcn --feedback_mode two_pass --d_gate 64 --topk 64 --lr 5e-4 --seed 0
    log "START triton install + bench"
    "$PY" -m pip install triton-windows --quiet --progress-bar off >> "$LOG" 2>&1 \
        && "$PY" bench_compile.py >> "$LOG" 2>&1 \
        || log "SKIP triton/bench (failed, non-blocking)"
    log "===== GPU line end ====="
}

cpu_line() {
    log "===== CPU line start ====="
    for S in 0 1 2 3 4; do
        log "START copy_mechanism seed $S"
        CUDA_VISIBLE_DEVICES=-1 "$PY" copy_mechanism.py --seed $S \
            > "results/mechanism/copy_mechanism_s${S}.out" 2>&1 \
            && log "DONE copy_mechanism seed $S" \
            || log "FAIL copy_mechanism seed $S (continue)"
    done
    log "START gate_source_test"
    CUDA_VISIBLE_DEVICES=-1 "$PY" gate_source_test.py \
        > "results/mechanism/gate_source_2x2.out" 2>&1 \
        && log "DONE gate_source_test" \
        || log "FAIL gate_source_test (continue)"
    log "===== CPU line end ====="
}

gpu_line &
GPU_PID=$!
cpu_line &
CPU_PID=$!
log "GPU PID=$GPU_PID | CPU PID=$CPU_PID"
wait $GPU_PID $CPU_PID
log "===== lines joined, writing report ====="

"$PY" - << 'PYEOF' > results/OVERNIGHT_R_STATUS.md 2>>results/overnight_r.log
import json
from pathlib import Path
from statistics import mean

print("# 夜间 R 队列执行报告\n")
print(f"- 完成时间: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}")
print("- 执行策略: 单项失败不中断；全部完成后关机（用户授权）\n")

print("## 1. d=384 规模复现（TinyStories；d256 对照: TF 15.2 / NG 16.5 / 2P 20.1）\n")
print("| 配置 | PPL | tokens/s |")
print("|---|---|---|")
for name in ('o384_tf', 'o384_ng', 'o384_2p'):
    p = Path(f'results/{name}/results.json')
    if p.exists():
        r = json.loads(p.read_text())
        tps = (r.get('budget') or {}).get('tokens_per_sec', 0)
        print(f"| {name} | {r['best_ppl']:.2f} | {tps:.0f} |")
    else:
        print(f"| {name} | 失败/未完成 | - |")

print("\n## 2. 公平 lr 的 20K 走势（WT；5K 对照: TF 214.5 / NG 176.0）\n")
p = Path('results/wt20k_tf_lr1e3_s0/results.json')
if p.exists():
    r = json.loads(p.read_text())
    print(f"- TF@1e-3 20K 步: {r['best_ppl']:.1f}")
else:
    print("- 失败/未完成")

print("\n## 3. lr 5e-4 细扫（WT；对照: NG@3e-4=176.0 / 2P@1e-3=201.5）\n")
for name in ('wt_ng_lr5e4_s0', 'wt_2p_lr5e4_s0'):
    p = Path(f'results/{name}/results.json')
    if p.exists():
        print(f"- {name}: {json.loads(p.read_text())['best_ppl']:.2f}")
    else:
        print(f"- {name}: 失败/未完成")

print("\n## 4. triton-windows / compile 基准\n")
tp = Path('results/analysis_v2/throughput_compile.json')
if tp.exists():
    t = json.loads(tp.read_text())
    for k in ('transformer', 'pcn_eager', 'pcn_compile'):
        if isinstance(t.get(k), (int, float)):
            print(f"- {k}: {t[k]:,.0f} tok/s")
    if 'pcn_compile_error' in t:
        print(f"- compile 失败: {t['pcn_compile_error'][:120]}")
else:
    print("- 未产出（见 overnight_r.log）")

print("\n## 5. 机制实验多 seed（7 变体 × 5 seed，学会判据 copy<1.0）\n")
rows = {}
for s in range(5):
    p = Path(f'results/mechanism/copy_mechanism_s{s}.json')
    if p.exists():
        for v in json.loads(p.read_text()):
            rows.setdefault(v['name'], []).append(v['curve'][-1]['copy_loss'])
print("| 变体 | n | copy 均值 | 学会率 |")
print("|---|---|---|---|")
for name, vals in rows.items():
    ok = sum(1 for x in vals if x < 1.0)
    print(f"| {name} | {len(vals)} | {mean(vals):.3f} | {ok}/{len(vals)} |")
if not rows:
    print("（未产出）")

print("\n## 6. 门来源 × 值通路 2×2 分解\n")
gp = Path('results/mechanism/gate_source_2x2.json')
if gp.exists():
    for v in json.loads(gp.read_text()):
        print(f"- {v['name']}: copy={v['copy_loss']:.3f} {'YES' if v['copy_loss'] < 1.0 else 'NO'}")
else:
    print("- 未产出（见 overnight_r.log）")

print("\n---\n完整日志: results/overnight_r.log | 系统即将关机（用户已授权）")
PYEOF

log "报告已生成 results/OVERNIGHT_R_STATUS.md"
log "全部完成，120 秒后关机（用户已授权）"
cmd //c "shutdown /s /t 120"
