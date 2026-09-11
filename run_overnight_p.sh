#!/bin/bash
# 夜间 P 队列（2026-09-10）：P1b 稳定矩阵（GPU）+ P3 门来源 2×2（CPU）
# 用户授权：完成后记录 -> 关机；单项失败不中断
cd "$(dirname "$0")"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=".venv/Scripts/python.exe"
LOG="results/overnight_p.log"
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $@" | tee -a "$LOG"; }

gpu_line() {
    log "===== GPU line: P1b stability matrix ====="
    BEST_CONF=""
    for K in 2 3; do
      for CLIP in 0.5 0.25; do
        for LRL in 1.5e-4 5e-4; do
          NAME="p1b_K${K}_c${CLIP}_l${LRL}"
          [ -f "results/$NAME/results.json" ] && { log "skip $NAME"; continue; }
          log "START $NAME"
          "$PY" train.py --exp_name $NAME --model pcn --no_gating --local_rule m1 \
            --n_pass $K --lr_loc $LRL --grad_clip $CLIP \
            --d_gate 64 --topk 64 --lr 5e-4 --batch_size 32 \
            --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 \
            --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed 0 >> "$LOG" 2>&1
          NAN=$(grep -c "NaN" "results/$NAME/train.log" 2>/dev/null || echo 999)
          PPL=$("$PY" -c "import json;print(json.load(open('results/$NAME/results.json'))['best_ppl'])" 2>/dev/null || echo 999)
          log "$NAME: NaN=$NAN best=$PPL"
          # 稳定标准：NaN 少且 PPL 合理（<35 为稳定完成的代理）
          if [ "$NAN" -lt 10 ] 2>/dev/null && [ "$(echo "$PPL < 35" | bc 2>/dev/null)" = "1" ]; then
            if [ -z "$BEST_CONF" ]; then
              BEST_CONF="$K $CLIP $LRL"
              log "FIRST STABLE CONF: K=$K clip=$CLIP lr_loc=$LRL (matrix continues)"
            fi
          fi
        done
      done
    done
    # 用第一个稳定配置补 m1-inf/m2-inf 终判（若找到）
    if [ -n "$BEST_CONF" ]; then
      set -- $BEST_CONF; BK=$1; BC=$2; BL=$3
      for MODE in m1 m2; do
        NAME="p1b_${MODE}inf_final"
        [ -f "results/$NAME/results.json" ] && { log "skip $NAME"; continue; }
        log "START $NAME (K=$BK clip=$BC lr_loc=$BL)"
        "$PY" train.py --exp_name $NAME --model pcn --no_gating --local_rule $MODE \
          --n_pass $BK --lr_loc $BL --grad_clip $BC \
          --d_gate 64 --topk 64 --lr 5e-4 --batch_size 32 \
          --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 \
          --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed 0 >> "$LOG" 2>&1
        log "DONE/checked $NAME"
      done
    else
      log "NO STABLE CONFIG — 代码级修复留白天（三选项已在 FINAL.json）"
    fi
    log "===== GPU line end ====="
}

cpu_line() {
    log "===== CPU line: P3 gate source 2x2 ====="
    CUDA_VISIBLE_DEVICES=-1 "$PY" gate_source_v2.py > results/mechanism/gate_source_v2.out 2>&1 \
      && log "P3 DONE" || log "P3 FAIL (continue)"
    log "===== CPU line end ====="
}

gpu_line &
GPU_PID=$!
cpu_line &
CPU_PID=$!
log "GPU PID=$GPU_PID CPU PID=$CPU_PID"
wait $GPU_PID $CPU_PID
log "===== lines joined, writing report ====="

"$PY" - << 'PYEOF' > results/OVERNIGHT_P_STATUS.md 2>>results/overnight_p.log
import json
from pathlib import Path
print("# 夜间 P 队列执行报告\n")
print(f"- 完成时间: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}\n")

print("## P1b 稳定矩阵（K x clip x lr_loc）\n")
print("| 配置 | NaN 数 | best PPL | 稳定 |")
print("|---|---|---|---|")
stable = []
for p in sorted(Path('results').glob('p1b_K*_c*_l*/results.json')):
    name = p.parent.name
    nan = 0
    tl = p.parent / 'train.log'
    if tl.exists():
        nan = tl.read_text(errors='ignore').count('NaN')
    ppl = json.loads(p.read_text())['best_ppl']
    ok = nan < 10 and ppl < 35
    if ok:
        stable.append(name)
    print(f"| {name} | {nan} | {ppl:.2f} | {'YES' if ok else 'no'} |")

print("\n## P1b 终判（若稳定配置存在）\n")
for mode in ('m1', 'm2'):
    p = Path(f'results/p1b_{mode}inf_final/results.json')
    if p.exists():
        r = json.loads(p.read_text())
        base = 16.50
        print(f"- {mode}-inf: PPL={r['best_ppl']:.2f} ({r['best_ppl']/base:.2f}x 基线；判据 m1<=19.0 / m2<=21.5)")
    else:
        print(f"- {mode}-inf: 未产出（无稳定配置）")

print("\n## P3 门来源 2x2（PCN 内部开关）\n")
gp = Path('results/mechanism/gate_source_v2.json')
if gp.exists():
    d = json.loads(gp.read_text())
    print(f"- 校准格 feat/wlat_e: {d['calibration']}  valid={d['valid']}")
    if d.get('valid'):
        for r in d['results']:
            print(f"- {r['name']}: mean={r['mean']} copies={r['copies']}")
else:
    print("- 未产出")
print("\n---\n完整日志: results/overnight_p.log | 系统即将关机（用户已授权）")
PYEOF

log "报告已生成"
log "全部完成，120 秒后关机（用户已授权）"
cmd //c "shutdown /s /t 120"
