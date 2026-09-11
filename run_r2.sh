#!/bin/bash
cd "$(dirname "$0")"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=".venv/Scripts/python.exe"
LOG="results/r2_supplement.log"
log() { echo "[$(date +%H:%M)] $@" | tee -a "$LOG"; }
run() { NAME=$1; shift; [ -f "results/$NAME/results.json" ] && { log "skip $NAME"; return; }
  log "START $NAME"; "$PY" train.py --exp_name "$NAME" "$@" >> "$LOG" 2>&1
  [ -f "results/$NAME/results.json" ] && log "DONE $NAME" || log "FAIL $NAME"; }

# W384 补 seed（n=3 -> 5）
for S in 3 4; do
  run "w384_tf_s$S" --dataset wikitext --model transformer --attn_impl sdpa --lr 1e-3 --layers 12 --d_model 384 --ffn_dim 1536 --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $S
  run "w384_ng_s$S" --dataset wikitext --model pcn --no_gating --d_gate 64 --topk 64 --lr 5e-4 --layers 12 --d_model 384 --ffn_dim 1536 --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $S
done
# decay 补 seed（n=1 -> 3，各自最优 lr 1e-3）
for S in 1 2; do
  run "decay_ts_lr1e3_s$S" --model transformer --attn_impl decay --lr 1e-3 --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $S
  run "decay_wt_lr1e3_s$S" --dataset wikitext --model transformer --attn_impl decay --lr 1e-3 --batch_size 16 --accum_steps 2 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps 500 --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed $S
done
# warmup 敏感性对照（500 已有；100/1500 两点）
for W in 100 1500; do
  run "warm${W}_tf_s0" --model transformer --attn_impl sdpa --lr 1e-3 --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps $W --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed 0
  run "warm${W}_ng_s0" --model pcn --no_gating --d_gate 64 --topk 64 --lr 5e-4 --grad_clip 0.5 --batch_size 32 --seq_len 256 --max_seq_len 256 --max_steps 5000 --warmup_steps $W --weight_decay 0.1 --eval_interval 1000 --log_interval 1000 --seed 0
done
log "R2 ALL DONE"
