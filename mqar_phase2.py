#!/usr/bin/env python3
"""MQAR 第二阶段：加大预算的对称复测（阶段一 2000 步/D64 下四臂均未解出，
仅门控臂 n=16 露头 0.10；MQAR 键值绑定难于纯复制，预算按比例放大）

阶段二协议：8000 步 × batch 64（数据量 8× 于阶段一），n=16，四臂对称同预算，
2 seed，解出即停。若门控解出而其余仍随机 → 标准化结论成立。
门控解出后追加 n∈{32,48}（其余臂维持 n=16 随机结论即可，不再加预算——
同预算下已失败，更高难度只会更差）。

用法：.venv/Scripts/python.exe mqar_phase2.py
"""
import os
for _v in ('HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE'):
    os.environ.setdefault(_v, '1')

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mqar_benchmark import train_one, infer_tput, OUT

STEPS2, BATCH2 = 8000, 64
# 承重臂（gate）2 seed；随机臂阶段一已证 2-seed 全平无斜率，阶段二 1 seed 即可
ARMS = [('tf', 1e-3, 'Transformer@1e-3', [0]),
        ('decay', 1e-3, 'TF-decay(SSM代理)@1e-3', [0]),
        ('ng', 3e-4, 'PCN no_gating@3e-4', [0]),
        ('gate', 3e-4, 'PCN 门控@3e-4', [0, 1])]


def main():
    print(f'===== MQAR 阶段二（{STEPS2}步 × b{BATCH2}，n=16，对称预算）=====\n')
    results = {'protocol': {'steps': STEPS2, 'batch': BATCH2, 'n': 16,
                            'seeds': {'gate': [0, 1], 'others': [0]},
                            'note': '阶段一(2000步/b32)仅 gate 在 n=16 达 0.10，其余随机且曲线平坦'},
               'arms': {}}
    gate_solved = False
    for arm, lr, label, seeds in ARMS:
        accs, curves = [], {}
        for s in seeds:
            c = train_one(arm, 16, s, lr, steps=STEPS2, batch=BATCH2)
            curves[f's{s}'] = c
            accs.append(c[-1]['acc'])
            print(f'  {label:<24} n=16 seed{s}: 终末 acc={c[-1]["acc"]:.3f} ({c[-1]["step"]} 步)')
        solved = all(a >= 0.9 for a in accs)
        results['arms'][arm] = {'label': label, 'n16_acc': accs, 'n16_solved': solved,
                                'curves': curves}
        if arm == 'gate' and solved:
            gate_solved = True
        print(f'    -> {"✅ 解出" if solved else "❌"}')

    # 门控解出 → 追加难度阶梯（1 seed）
    if gate_solved:
        print('\n----- 门控臂难度阶梯 -----')
        results['gate_ladder'] = {}
        for n in (32, 48):
            c = train_one('gate', n, 0, 3e-4, steps=STEPS2, batch=BATCH2)
            acc = c[-1]['acc']
            results['gate_ladder'][f'n{n}'] = {'acc': acc, 'solved': acc >= 0.9,
                                               'curves': {'s0': c}}
            print(f'  PCN 门控 n={n} seed0: 终末 acc={acc:.3f} ({c[-1]["step"]} 步)')
        results['gate_max_n'] = max(
            [n for n in (16, 32, 48)
             if (n == 16 and results['arms']['gate']['n16_solved'])
             or (f'n{n}' in results.get('gate_ladder', {})
                 and results['gate_ladder'][f'n{n}']['solved'])] + [0])

    json.dump(results, open(OUT / 'mqar_phase2.json', 'w'), indent=2)
    print(f'\n输出: {OUT}/mqar_phase2.json')


if __name__ == '__main__':
    main()
