#!/usr/bin/env python3
"""R1 — 规律曲线的 bootstrap 95% CI 计算（审稿 W4 响应）

对每个曲线点：均值、bootstrap 95% CI（B=1000）、样本量 n、
效应方向稳健性（NG 均值是否优于 TF 全部/多数 seed）。
输出：results/analysis_v2/scale_curve_with_ci.json
"""
import sys, json, random
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))


def ppl(names):
    v = []
    for n in names:
        p = Path(f'results/{n}/results.json')
        if p.exists():
            v.append(json.loads(p.read_text())['best_ppl'])
    return v


def bootstrap_ci(vals, B=1000, alpha=0.05):
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (None, None)
    rng = random.Random(42)
    boots = []
    for _ in range(B):
        boots.append(mean(rng.choices(vals, k=len(vals))))
    boots.sort()
    return (boots[int(B*alpha/2)], boots[int(B*(1-alpha/2))])


AUTH = {
 'WT d256@164M': (['wt20k_tf_lr1e3_s0'], ['wt20k_causal_nogate_s0']),
 'TS d256@41M': (['ts_sdpa_tf_lr1e3_s0'] + [f'ts_tf_lr1e3_s{i}' for i in (1,2,3,4)],
                 [f'ts_causal_no_gating_s{i}' for i in range(5)]),
 'WT d256@41M': (['wt_causal_tf_lr1e3_s0'] + [f'wt_tf_lr1e3_s{i}' for i in (1,2,3,4)],
                 [f'wt_causal_no_gating_s{i}' for i in range(5)]),
 'TS d384@41M': (['o384_tf'] + [f'o384_tf_s{i}' for i in (1,2,3,4)],
                 ['o384_ng'] + [f'o384_ng_s{i}' for i in (1,2,3,4)]),
 'WT d384@41M': ([f'w384_tf_s{i}' for i in range(5)],
                 [f'w384_ng_s{i}' for i in range(5)]),
 'TS d512@41M': ([f's512_tf_s{i}' for i in range(3)], [f's512_ng_s{i}' for i in range(3)]),
 'WT d512@41M': ([f'w512_tf_s{i}' for i in range(5)], [f'w512_ng_s{i}' for i in range(5)]),
 # D1 数据量切片（同 d256，数据量轴——二维规律的第二个切片）
 'TS d256@2.5M': ([f'dslice_tf_2.5M_s{i}' for i in (0,1)], [f'dslice_ng_2.5M_s{i}' for i in (0,1)]),
 'TS d256@5M': ([f'dslice_tf_5M_s{i}' for i in (0,1)], [f'dslice_ng_5M_s{i}' for i in (0,1)]),
 'TS d256@10M': ([f'dslice_tf_10M_s{i}' for i in (0,1)], [f'dslice_ng_10M_s{i}' for i in (0,1)]),
 'TS d256@20M': ([f'dslice_tf_20M_s{i}' for i in (0,1)], [f'dslice_ng_20M_s{i}' for i in (0,1)]),
}


def main():
    out = []
    print(f'{"point":<15} {"n":>6} {"TF mean [CI]":>26} {"NG mean [CI]":>26} {"adv%":>7} {"方向":>6}')
    for pt, (tfn, ngn) in AUTH.items():
        tfv, ngv = ppl(tfn), ppl(ngn)
        if not tfv or not ngv:
            print(f'{pt:<15}  MISSING'); continue
        tfm, ngm = mean(tfv), mean(ngv)
        tfci, ngci = bootstrap_ci(tfv), bootstrap_ci(ngv)
        adv = (1 - ngm/tfm) * 100
        robust = '全胜' if max(ngv) < min(tfv) else ('多数' if ngm < tfm else '负')
        out.append({'point': pt, 'tf': {'mean': round(tfm,2), 'ci95': [round(tfci[0],2), round(tfci[1],2)], 'n': len(tfv)},
                    'ng': {'mean': round(ngm,2), 'ci95': [round(ngci[0],2), round(ngci[1],2)], 'n': len(ngv)},
                    'adv_pct': round(adv,1), 'robustness': robust})
        print(f'{pt:<15} {len(tfv)}/{len(ngv):>2} {tfm:>9.2f} [{tfci[0]:>7.2f},{tfci[1]:>7.2f}]'
              f' {ngm:>9.2f} [{ngci[0]:>7.2f},{ngci[1]:>7.2f}] {adv:>+6.1f}% {robust:>5}')
    Path('results/analysis_v2').mkdir(exist_ok=True)
    json.dump(out, open('results/analysis_v2/scale_curve_with_ci.json', 'w'), indent=2)
    print('\nsaved -> results/analysis_v2/scale_curve_with_ci.json')


if __name__ == '__main__':
    main()
