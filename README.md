# PRISM V4-V6

**Predictive Coding Networks in Language Modeling: Sample Efficiency, Mechanistic Unreachability, and Measurement Discipline — A Single-GPU Architecture Study**

> 预测编码网络在语言建模中的样本效率、机制不可达性与测量纪律：一项单卡架构研究
>
> Author: Keyu Deng (Independent Researcher) | Preprint: [AiraXiv:2609.0010](https://airaxiv.com/papers/view/2609.0010/) | Hardware: RTX 4080 16GB, PyTorch 2.14

---

## Core Findings

1. **Two-Dimensional Asymmetric Law**: The error-stream variant (no_gating) advantage over Transformers follows different shapes along two orthogonal axes:
   - **Data-reduction axis** (fixed model, decreasing data): monotonically increases to **+84%**
   - **Parameter-increase axis** (fixed data, increasing parameters): inverted-U, peak at ≈1000 tokens/parameter (**+60.8%** on WikiText-103)
   - Deployment implication (v2.2 rescoped): the advantage cashes out in **cross-domain / cold-start adaptation**, not in-domain fine-tuning — see finding 5

2. **Compositional Property with Symmetric Unreachability**: PCN's exact-copy capability (15/15 vs Transformer's 0/30) survives component removal but cannot be transplanted into standard blocks — confirmed across a complete 2×2×2 factorial design (8 cells, all successful) plus 5 standard-block graft attempts (all failed). The weight-magnitude constraint during fine-tuning (ΔW ≈ TF/2) follows the same pattern (second emergent property, robust to fair baselines at 1.91×) — reframed as **parameter economy**, not forgetting protection.

3. **Local Learning Rule Boundary**: Whittington-Bogacz predictive coding rules exactly match backpropagation on MLPs (loss 0.0002 vs 0.0002) but exhibit a structural 2.4-3.7× gap on PCN — the equivalence does not cross the structural complexity boundary. V6 closed this line per its failure exit.

4. **Measurement Discipline**: Four systematic measurement biases (non-causal leakage, suboptimal baseline hyperparameters — including one V6-series recurrence with errata, invalid custom controls, synthetic-benchmark frequency shortcut) were each captured by pre-registered verification procedures — codified into 10 reusable lessons (see `LESSONS.md`).

5. **Terminal Adaptation Boundaries (v2.2)**: The adaptation advantage is real but strictly **cross-domain**: one-shot user adaptation +58.5% vs fair TF +5.4%; true streaming (prequential, one pass) +24.0% vs TF −80.9%. In-domain personalization has **no headroom** (24-cell calibration grid, all negative for both archs); at matched adaptation PCN forgets *more* (plasticity–stability tradeoff). Deployment rule: adapt on cold start, freeze when converged. Terminal efficiency completed: CUDA-Graph b1 throughput 102% of TF with −28% energy; int4 recipe (g64 + W_res kept fp16) near-lossless — W_res (non-additive residual trunk) is PCN's entire quantization fragility. MQAR is a **negative** result (all four archs at chance at toy scale; copy advantage does not extend to key-value binding).

## Repository Structure

```
PRISM V4/
├── train.py                    # Unified training script (all architectures & ablations)
├── src/
│   ├── models/
│   │   ├── pcn.py              # PCN architecture (with bmm gate, K-pass, ablation switches)
│   │   └── transformer.py      # Transformer baseline (SDPA/MHA/decay variants)
│   ├── data/
│   │   ├── tinystories.py      # TinyStories loader (HF dataset, cached)
│   │   └── wikitext.py         # WikiText-103 loader (cached to disk)
│   └── analysis/
│       └── cka.py              # CKA (Centered Kernel Alignment)
├── run_*.sh                    # Reproducible experiment pipelines (idempotent)
├── analyze_v2.py               # Behavioral signatures (CKA, gate entropy, error-surprise)
├── ci_bootstrap.py             # Bootstrap 95% CI for scale-data curve
├── copy_mechanism.py           # Copy task mechanism experiments (7 variants)
├── gate_source_v2.py           # Gate source × value pathway 2×2 decomposition
├── block_ablation.py           # Block sub-component ablation (FFN, update path, etc.)
├── factorial_fill.py           # 2×2×2 full factorial design
├── diagnose.py                 # Zero-training diagnosis (gradient flow, gate health)
├── toy_test.py                 # Toy task learnability matrix
├── copy_bench.py               # Copy task formal benchmark (multi-length × multi-seed)
├── bench_t1a.py                # Gate bmm vs loop throughput benchmark
├── bench_infer.py              # Inference benchmark (batch 1-32)
├── bench_int8.py               # int8 quantization sensitivity
├── bench_energy.py             # T4: NVML energy (J/token, train+infer)
├── bench_b1_graph.py           # T5: batch1 CUDA Graph optimization
├── bench_int4.py               # T6: int4 ladder + W_res probe
├── forgetting_benchmark.py     # Catastrophic forgetting (calibrated/aggressive)
├── forgetting_calibration.py   # Per-arch adaptation budget calibration
├── skeleton_fair_recheck.py    # Weight-constraint fair-baseline recheck
├── mqar_benchmark.py           # MQAR phase 1 (+ phase2/3/4 scripts)
├── streaming_demo.py           # Prequential streaming adaptation demo
├── local_rule_test.py          # W&B local rule Level 0 (MLP calibration)
├── local_rule_level2.py        # W&B local rule Level 2 (LM task)
├── LESSONS.md                  # 10 lessons from measurement failures
├── GOAL.md                     # Research roadmap and current status
├── V4_PLAN.md                  # Original V4 design document
└── results/
    ├── analysis_v2/            # Scale curve with CI, signatures, efficiency data
    ├── mechanism/              # Copy, factorial, skeleton, forgetting, MQAR, local rule
    ├── efficiency/             # Throughput, inference, int8/int4, energy, graphs
    ├── demo/                   # Personalization + streaming demos
    ├── PRISM_V5_TECH_REPORT.md # Full technical report (Chinese, v2.2)
    ├── V6_CLOSURE.md           # V6 final closure document
    ├── SUBMISSION_PACKAGE.md   # AiraXiv submission metadata
    └── EFFICIENCY_REPORT.md    # Efficiency results (T1-T6)
```

## Quick Start

### Environment

```bash
# Python 3.12+ with CUDA-capable PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets numpy

# Or use requirements.txt
pip install -r requirements.txt
```

### Smoke Test (2 minutes)

```bash
# Verify environment: 10-step dry run on TinyStories
python train.py --model pcn --no_gating --d_gate 64 --topk 64 \
    --lr 5e-4 --batch_size 32 --seq_len 256 --max_seq_len 256 \
    --max_steps 10 --warmup_steps 5 --eval_interval 10 --log_interval 5 \
    --exp_name _smoke_test

# Clean up
rm -rf results/_smoke_test
```

### Data Setup

```bash
# First-time download (uses hf-mirror.com for China network; see tinystories.py)
python -c "
from src.data.tinystories import get_dataloaders
tr, va, tok = get_dataloaders(seq_len=256, batch_size=4, max_train=100, max_val=100)
print('OK', len(tr), len(va))
"
```

**⚠️ China network note**: `tinystories.py` sets `HF_ENDPOINT=https://hf-mirror.com` and offline mode by default. For first download, override:
```bash
HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python -c "..."
```

### Reproduce Key Results

```bash
# Scale-data ratio curve (core finding #1)
bash run_ts_causal_seeds.sh       # TinyStories multi-seed
bash run_wt_causal.sh             # WikiText-103 multi-seed
python ci_bootstrap.py            # Generate CI table

# Copy task mechanism (core finding #2)
python copy_mechanism.py --seed 0  # 7-variant matrix
python factorial_fill.py           # 2×2×2 full factorial

# Efficiency benchmarks (v2 improvement)
python bench_t1a.py               # bmm vs loop throughput
python bench_infer.py             # batch 1-32 inference

# V6 finale series (v2.2)
python forgetting_benchmark.py calibrated   # forgetting, matched-adaptation regime
python bench_energy.py            # T4 energy (needs nvidia-ml-py)
python bench_b1_graph.py          # T5 batch1 CUDA Graph
python bench_int4.py              # T6 int4 + W_res probe
python streaming_demo.py          # prequential streaming adaptation
python mqar_benchmark.py          # MQAR (negative result, phase 1)
```

## Results File Schema

| File | Content | Key fields |
|---|---|---|
| `analysis_v2/scale_curve_with_ci.json` | 11-point curve with bootstrap 95% CI | `point, tf.mean, tf.ci95, ng.mean, ng.ci95, adv_pct, robustness` |
| `mechanism/copy_mechanism_s*.json` | Copy task 7 variants × 5 seeds | `name, curve, gate_std` |
| `mechanism/factorial_fill.json` | 2×2×2 factorial (3 missing cells filled) | `feat_e_ffn, qk_e_ffn, feat_h_ffn` |
| `mechanism/local_rule_level2_FINAL.json` | W&B local rule Level 2 verdict | `p1a.lr_scan, p1b_final` |
| `efficiency/t1a_bench.json` | bmm vs loop throughput | `tf, pcn_bmm, pcn_loop, speedup, pct_of_tf` |
| `efficiency/t3_infer.json` | Inference batch 1-32 | `1, 4, 16, 32` × `tf, no_gating, gate_bmm` |
| `mechanism/forgetting_benchmark_*.json` | Forgetting (calibrated/aggressive, 4 seeds) | `raw, summary` |
| `mechanism/skeleton_fair_recheck.json` | ΔW constraint fair-baseline check | `tag, dw, ppl_b/a, ch` |
| `demo/streaming_demo.json` | Prequential streaming | `traj, frozen_ppl, online_gain_pct` |
| `efficiency/t4_energy.json` | Energy J/token (train + b1/b32 infer) | `train, infer, to_ppl_ts5k` |
| `efficiency/t5_b1_graph.json` | CUDA Graph b1 optimization | `eager, graph, graph_speedup, max_logit_diff` |
| `efficiency/t6_int4.json` | int4 ladder + W_res probe | `methods, size_mb, wres_probe` |

## Known Issues & Environment Notes

1. **HF offline variables must be exported in shell**, not just `setdefault` in Python (datasets 5.x quirk)
2. **torch.compile causes negative optimization** for the gate loop path (18.7K vs eager 28.4K tok/s) — fixed by bmm restructuring (161K tok/s)
3. **Windows triton-windows** installed but `torch.compile` still fails for custom kernels — the bmm math equivalence makes Triton unnecessary
4. **Evaluation alignment**: dataset already shifts (input=tokens[:-1], target=tokens[1:]) — use full-tensor CE, NOT `logits[:,:-1]` vs `y[:,1:]` (double-shift bug, caught 3 times)

## License

MIT License (see LICENSE file)

## Citation

```bibtex
@preprint{deng2026prism,
  title={预测编码网络在语言建模中的样本效率、机制不可达性与测量纪律：一项单卡架构研究},
  author={Keyu Deng},
  year={2026},
  note={AiraXiv:2609.0010}
}
```
