#!/usr/bin/env python3
"""analysis_stats — A2：审稿回应的统计补强（零训练，纯分析）

1. run_index.csv：全部 run 的元数据索引（审稿建议 5：可复现性卫生）
2. 配对统计：3-seed 关键对比（HYB vs TF / PCN）的逐 seed 差值 + bootstrap 95% CI
   + 双样本 t 检验（审稿问题 Q1）
3. Power 计算：给定观测方差，检测 10% PPL 差异所需 seed 数（审稿建议 5）

用法：.venv/Scripts/python.exe analysis_stats.py
"""
import csv, json, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results'
OUT_JSON = RESULTS / 'v7' / 'analysis_stats.json'
OUT_CSV = RESULTS / 'run_index.csv'


# ---------------- 1. run_index.csv ----------------

def build_run_index():
    rows = []
    for rj in sorted(RESULTS.glob('*/results.json')):
        try:
            r = json.load(open(rj))
            cfg = r.get('config', {})
            rows.append({
                'exp_name': rj.parent.name,
                'model': cfg.get('model'),
                'dataset': cfg.get('dataset'),
                'lr': cfg.get('lr'),
                'lr_backbone': cfg.get('lr_backbone', ''),
                'layers': cfg.get('layers'),
                'tf_layers': cfg.get('tf_layers', ''),
                'd_model': cfg.get('d_model'),
                'seed': cfg.get('seed'),
                'amp_dtype': cfg.get('amp_dtype', ''),
                'best_ppl': round(r.get('best_ppl', float('nan')), 2),
                'nan_recoveries': r.get('nan_recoveries', ''),
                'time_min': round(r.get('total_time_s', 0) / 60, 1),
                'n_params_M': round(r.get('n_params', 0) / 1e6, 1),
                'results_path': str(rj.relative_to(ROOT)).replace('\\', '/'),
            })
        except Exception as e:
            rows.append({'exp_name': rj.parent.name, 'error': str(e)})
    cols = ['exp_name', 'model', 'dataset', 'lr', 'lr_backbone', 'layers',
            'tf_layers', 'd_model', 'seed', 'amp_dtype', 'best_ppl',
            'nan_recoveries', 'time_min', 'n_params_M', 'results_path']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in cols})
    return rows


# ---------------- 2. 配对统计 ----------------

def paired_bootstrap(x, y, n_boot=20000, seed=0):
    """逐对差值的 bootstrap 95% CI（x[i]-y[i] 按同 seed 配对）"""
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(x, y)]
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    return diffs, statistics.mean(diffs), lo, hi


def welch_t(x, y):
    """双样本 Welch t（小样本近似，报告 t 值与近似 p）"""
    nx, ny = len(x), len(y)
    vx, vy = statistics.variance(x), statistics.variance(y)
    t = (statistics.mean(x) - statistics.mean(y)) / math.sqrt(vx / nx + vy / ny)
    # 正态近似 p（n=3 时仅供量级参考）
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
    return t, p


def do_paired_stats():
    sr = json.load(open(RESULTS / 'v7' / 'seed_replication.json'))
    out = {}

    def compare(name, xa, ya, unit):
        diffs, mean_d, lo, hi = paired_bootstrap(xa, ya)
        t, p = welch_t(xa, ya)
        out[name] = {'per_seed_x': xa, 'per_seed_y': ya,
                     'paired_diffs': [round(d, 3) for d in diffs],
                     'mean_diff': round(mean_d, 3),
                     'boot95_ci': [round(lo, 3), round(hi, 3)],
                     'welch_t': round(t, 2), 'approx_p': f'{p:.2e}',
                     'unit': unit}
        print(f'  {name}: 逐seed差 {["%.3f" % d for d in diffs]} '
              f'| 均差 {mean_d:+.3f} [{lo:+.3f}, {hi:+.3f}] '
              f'| t={t:.1f} p≈{p:.1e}')

    print('--- 配对统计（seed 0/1/2）---')
    compare('pretrain_HYB_vs_TF (PPL, 负=HYB优)',
            sr['pretrain']['hyb']['per_seed'], sr['pretrain']['tf']['per_seed'], 'PPL')
    compare('pretrain_HYB_vs_PCN (PPL, 负=HYB优)',
            sr['pretrain']['hyb']['per_seed'], sr['pretrain']['pcn']['per_seed'], 'PPL')
    compare('coldstart_HYB_vs_TF (pp)',
            [v for v in sr['coldstart']['hyb']['per_seed']],
            [v for v in sr['coldstart']['tf']['per_seed']], 'pp')

    # 流式：按 seed 配对（同 lr=5e-5）
    st = {}
    for k, v in sr['streaming'].items():
        st[k] = v['online_gain_pct']
    hyb = [st[f'hyb_s{s}_lr5e-05'] for s in (0, 1, 2)]
    tf = [st[f'tf_s{s}_lr5e-05'] for s in (0, 1, 2)]
    pcn = [st[f'pcn_s{s}_lr5e-05'] for s in (0, 1, 2)]
    compare('stream@5e-5_HYB_vs_TF (pp)', hyb, tf, 'pp')
    compare('stream@5e-5_HYB_vs_PCN (pp)', hyb, pcn, 'pp')
    return out


# ---------------- 3. Power 计算 ----------------

def power_table():
    """两样本双侧 α=0.05、power=0.8：每组 n ≈ 15.68·σ²/Δ²"""
    rows = []
    print('\n--- Power：检测 Δ 所需每组 seed 数（α=0.05, power=0.8）---')
    print('  {:28s} σ      Δ(10%级)   n/组'.format('对比'))
    for name, sigma, ref in [
            ('预训练 PPL (TF 观测σ)', 2.62, 99.34),
            ('预训练 PPL (HYB 观测σ)', 0.22, 63.18),
            ('单发改善 (HYB 观测σ)', 4.3, 60.3)]:
        delta = 0.10 * ref
        n = math.ceil(15.68 * sigma ** 2 / delta ** 2)
        rows.append({'comparison': name, 'sigma': sigma,
                     'delta_10pct': round(delta, 2), 'n_per_group': n})
        print(f'  {name:28s} {sigma:5.2f}  {delta:7.2f}    {n}')
    return rows


def main():
    rows = build_run_index()
    print(f'run_index: {len(rows)} 个 run → {OUT_CSV.name}')
    stats = do_paired_stats()
    power = power_table()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'paired_stats': stats, 'power': power,
               'n_runs_indexed': len(rows)},
              open(OUT_JSON, 'w'), indent=2, ensure_ascii=False)
    print(f'\n输出: {OUT_JSON} + {OUT_CSV}')


if __name__ == '__main__':
    main()
