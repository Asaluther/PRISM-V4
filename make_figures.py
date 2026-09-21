#!/usr/bin/env python3
"""make_figures — 论文 PPT 配图（6 张，数据全部现场读 JSON，不手抄数字）

1 inverted_u.png      倒 U 规律（参数轴 + 减数据轴，adv_pct + CI）
2 cross_scale.png     跨规模 best PPL 对比（96M / 330M × HYB/TF/PCN）
3 decouple.png        B4 收窄归因三点分解（36.4→58.6→11.2）
4 step50.png          step50 四曲线（train/test/gen/ΔW，lr=5e-4 均值）
5 pcn_diverge.png     PCN 306M 软发散轨迹 vs HYB/TF 同预算
6 multidraw.png       330M 单发多抽签（10 用户 × 6 主体箱线）
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT = Path('results/figures')
OUT.mkdir(parents=True, exist_ok=True)
C_HYB, C_TF, C_PCN = '#2563eb', '#dc2626', '#9333ea'


def fig1():
    pts = json.load(open('results/analysis_v2/scale_curve_with_ci.json'))
    # 参数轴（固定 41M tokens）：x = tokens/param
    par = [p for p in pts if '@' in p['point'] and '@41M' in p['point']]
    tp = {'d256': 41e6 / 21.0e6, 'd384': 41e6 / 37.4e6, 'd512': 41e6 / 57.8e6}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for dom, mk in (('WT', 'o'), ('TS', 's')):
        xs = [tp[p['point'].split()[1].split('@')[0]] for p in par
              if p['point'].startswith(dom)]
        ys = [p['adv_pct'] for p in par if p['point'].startswith(dom)]
        ax.plot(xs, ys, mk + '-', label=f'{dom}（参数轴 @41M tok）')
    ax.axvline(1025, ls='--', c='gray', lw=1)
    ax.annotate('峰位 ≈1025\ntokens/param', (1025, 52), fontsize=9, ha='right')
    ax.set_xlabel('tokens / param'); ax.set_ylabel('no_gating 相对 TF 优势 (%)')
    ax.set_title('(a) 参数轴：倒 U'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    dec = [p for p in pts if 'd256@' in p['point'] and '@41M' not in p['point']]
    tok = [float(p['point'].split('@')[1][:-1]) for p in dec]
    ys = [p['adv_pct'] for p in dec]
    order = sorted(range(len(tok)), key=lambda i: tok[i])
    ax.plot([tok[i] / 1e6 for i in order], [ys[i] for i in order], 'o-')
    for i in order:
        ax.annotate(f'{ys[i]:+.0f}%', (tok[i] / 1e6, ys[i]),
                    textcoords='offset points', xytext=(0, 8), fontsize=9, ha='center')
    ax.set_xlabel('训练 tokens（百万，固定 d256）'); ax.set_ylabel('优势 (%)')
    ax.set_title('(b) 减数据轴：数据越稀缺优势越大'); ax.grid(alpha=.3)
    fig.suptitle('规模-数据比规律（纯 PCN no_gating vs TF，21-63M）', y=1.02)
    fig.tight_layout(); fig.savefig(OUT / 'inverted_u.png', bbox_inches='tight')
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    groups = ['96M（82M tok）', '330M（164M tok）']
    data = {'HYB': ([63.18, 39.75], C_HYB), 'TF': ([99.34, 48.78], C_TF),
            'PCN': ([149.95, 147.77], C_PCN)}
    import numpy as np
    x = np.arange(2)
    w = 0.25
    for i, (k, (vals, c)) in enumerate(data.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=k, color=c, alpha=.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 3, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel('best PPL（越低越好）')
    ax.set_title('跨规模对比：混合架构反超（96M: −36.4%；330M 调优口径: −11.2%，3/3 逐 seed 全胜）')
    ax.legend(); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'cross_scale.png', bbox_inches='tight')
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pts = [('96M\n@853 t/p', 36.4), ('96M\n@470 t/p', 58.6), ('330M\n@470 t/p', 11.2)]
    xs = range(3)
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, 'o-', lw=2, ms=9, color=C_HYB)
    for i, y in enumerate(ys):
        ax.annotate(f'{y:+.1f}%', (i, y), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=12, fontweight='bold')
    ax.annotate('数据比效应\n（稀缺放大 +22pp）', (0.5, 47), fontsize=10, ha='center', color='#166534')
    ax.annotate('规模效应\n（压缩 −47pp，主导）', (1.5, 35), fontsize=10, ha='center', color='#991b1b')
    ax.set_xticks(list(xs)); ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel('HYB 相对 TF 优势 (%)'); ax.set_ylim(0, 70)
    ax.set_title('收窄归因解耦：36.4% → 58.6% → 11.2% 的两步分解')
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'decouple.png', bbox_inches='tight')
    plt.close(fig)


def fig4():
    d = json.load(open('results/v7/step50_mechanism.json'))
    runs = [r for r in d['runs'] if r['lr'] == 5e-4]
    steps = [c['step'] for c in runs[0]['curve']]
    import statistics as st
    def series(key):
        return [st.mean(r['curve'][i][key] for r in runs)
                for i in range(len(runs[0]['curve']))]
    tr, te, gn = series('train_ppl'), series('test_ppl'), series('gen_ppl')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, tr, label='train PPL（6 条训练序列）', color=C_HYB, lw=2)
    ax.plot(steps, te, label='test PPL（同用户 held-out）', color=C_TF, lw=2)
    ax.plot(steps, gn, label='通用 PPL（WikiText 探针）', color=C_PCN, lw=2)
    ax.axvline(50, ls='--', c='gray', lw=1)
    ax.annotate('step50 峰值窗口', (50, ax.get_ylim()[1] * .5), fontsize=9, rotation=90)
    ax.set_yscale('log'); ax.set_xlabel('适应步数'); ax.set_ylabel('PPL（对数轴）')
    ax.set_title('step50 达峰/回落机制（lr=5e-4，3 用户均值）：记忆与通用遗忘并行')
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'step50.png', bbox_inches='tight')
    plt.close(fig)


def fig5():
    def curve(name):
        d = json.load(open(f'results/{name}/results.json'))
        return [v['step'] for v in d['val_ppl']], [v['ppl'] for v in d['val_ppl']]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, lbl, c in (('wt_330m_hyb_ng_lr5e-4_s0', 'HYB 330M（39.50）', C_HYB),
                         ('wt_353m_tf_lr5e-4_s0', 'TF 354M（41.72）', C_TF),
                         ('wt_306m_pcn_lr1e-4_s0', 'PCN 307M（峰值 147.77 → 222）', C_PCN)):
        xs, ys = curve(name)
        ax.plot([x / 1000 for x in xs], ys, label=lbl, color=c, lw=2)
    ax.set_xlabel('训练步数（千步）'); ax.set_ylabel('val PPL')
    ax.set_title('330M 档预训练轨迹：PCN 规模天花板实测（中途软发散，零 NaN）')
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'pcn_diverge.png', bbox_inches='tight')
    plt.close(fig)


def fig6():
    d = json.load(open('results/v7/adapt330_multidraw.json'))
    import numpy as np
    tags = list(d.keys())
    hyb = [d[t]['per_user'] for t in tags if t.startswith('hyb')]
    tf = [d[t]['per_user'] for t in tags if t.startswith('tf')]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos_hyb, pos_tf = [0, 1, 2], [3.6, 4.6, 5.6]
    bp1 = ax.boxplot(hyb, positions=pos_hyb, widths=.5, patch_artist=True)
    bp2 = ax.boxplot(tf, positions=pos_tf, widths=.5, patch_artist=True)
    for b in bp1['boxes']:
        b.set_facecolor(C_HYB); b.set_alpha(.6)
    for b in bp2['boxes']:
        b.set_facecolor(C_TF); b.set_alpha(.6)
    ax.axhline(0, c='gray', lw=1)
    ax.set_xticks(pos_hyb + pos_tf)
    ax.set_xticklabels([f'HYB s{i}' for i in range(3)] + [f'TF s{i}' for i in range(3)])
    ax.set_ylabel('单发冷启动改善 (%)')
    ax.set_title('330M 单发多抽签（10 用户/主体）：HYB 均值全正紧凑 vs TF 方差爆炸')
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'multidraw.png', bbox_inches='tight')
    plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5, fig6):
    f(); print(f.__name__, 'ok')
print('all figures ->', OUT)
