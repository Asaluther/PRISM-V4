#!/usr/bin/env python3
"""PRISM V4 — 分析脚本（PPL 曲线、误差分布、CKA）"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_ppl_curves(model_dirs, save_path):
    """PPL 收敛曲线对比"""
    plt.figure(figsize=(10, 6))
    for path, label in model_dirs:
        results = json.load(open(Path(path) / 'results.json'))
        steps = [d['step'] for d in results['val_ppl']]
        ppls = [d['ppl'] for d in results['val_ppl']]
        plt.plot(steps, ppls, 'o-', label=label, markersize=4)

    plt.xlabel('Steps')
    plt.ylabel('Validation PPL')
    plt.title('PPL Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PPL curve saved: {save_path}")


def plot_error_heatmap(model_dir, save_path):
    """PCN 预测误差空间分布热力图"""
    # 如果有保存的误差数据就画，否则提示
    results_path = Path(model_dir) / 'results.json'
    if not results_path.exists():
        print(f"  No results found at {results_path}")
        return

    results = json.load(open(results_path))
    if 'error_norms' not in results:
        print(f"  No error_norms data in results (need return_stats=True during training)")
        return

    error_data = np.array(results['error_norms'])
    if len(error_data.shape) == 2:
        # [layers, positions]
        plt.figure(figsize=(12, 6))
        im = plt.imshow(error_data, aspect='auto', cmap='hot', interpolation='nearest')
        plt.xlabel('Position')
        plt.ylabel('Layer')
        plt.title('PCN Prediction Error Norms (Layer × Position)')
        plt.colorbar(im, label='Error Norm')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Error heatmap saved: {save_path}")


def summary_table(model_dirs):
    """打印结果汇总表"""
    print(f"\n{'='*70}")
    print(f"{'Model':<30} {'Best PPL':>10} {'Params':>10} {'Steps':>10}")
    print(f"{'='*70}")
    for path, label in model_dirs:
        results_path = Path(path) / 'results.json'
        if results_path.exists():
            results = json.load(open(results_path))
            ppl = results.get('best_ppl', 'N/A')
            params = results.get('n_params', 'N/A')
            if isinstance(params, (int, float)):
                params = f"{params/1e6:.1f}M"
            steps = len(results.get('val_ppl', []))
            if steps > 0:
                steps = results['val_ppl'][-1]['step']
            print(f"  {label:<28} {str(ppl):>10} {str(params):>10} {str(steps):>10}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='PRISM V4 Analysis')
    parser.add_argument('--models', nargs='+', required=True,
                       help='model_dirs in format dir:label (e.g., results/pcn:PCN results/transformer:Transformer)')
    parser.add_argument('--output_dir', type=str, default='./analysis/')
    args = parser.parse_args()

    model_dirs = []
    for m in args.models:
        parts = m.split(':', 1)
        model_dirs.append((parts[0], parts[1] if len(parts) > 1 else parts[0]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PRISM V4 Analysis")
    print(f"Models: {len(model_dirs)}")

    # 汇总表
    summary_table(model_dirs)

    # PPL 曲线
    plot_ppl_curves(model_dirs, output_dir / 'ppl_curves.png')

    # PCN 误差热力图
    for path, label in model_dirs:
        if 'pcn' in label.lower():
            plot_error_heatmap(path, output_dir / f'error_heatmap_{label.replace(" ", "_")}.png')

    print(f"\n分析完成，输出到: {output_dir}")


if __name__ == '__main__':
    main()
