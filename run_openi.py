#!/usr/bin/env python3
"""启智社区（OpenI）训练任务启动脚本

自动处理：
1. 安装依赖（transformers + datasets）
2. 从挂载路径复制 WikiText 数据到 cache/
3. 启动 200M 参数训练（PCN 或 Transformer）

启动文件设为 run_openi.py，运行参数可加：
  --mode pcn     （默认）跑 PCN 200M
  --mode tf      跑 Transformer 200M
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path


def install_deps():
    """安装必要依赖"""
    deps = ['transformers', 'datasets']
    for dep in deps:
        try:
            __import__(dep)
            print(f'[依赖] {dep} 已安装')
        except ImportError:
            print(f'[依赖] 安装 {dep}...')
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', dep, '-q'],
            )


def setup_data():
    """把数据集从挂载路径复制到代码期望的 cache/ 目录"""
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)

    # 启智数据集挂载路径（可能的位置）
    possible_sources = [
        Path('/dataset'),
        Path('/datasets'),
        Path(os.environ.get('DATASET_PATH', '')),
    ]

    # 也搜索当前目录下的数据集文件
    for root, dirs, files in os.walk('/'):
        # 限制搜索深度，避免太慢
        depth = root.count('/')
        if depth > 3:
            dirs[:] = []
            continue
        for f in files:
            if f in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
                possible_sources.append(Path(root) / f)

    # 查找并复制
    for name in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
        target = cache_dir / name
        if target.exists():
            print(f'[数据] {name} 已存在')
            continue

        found = False
        for src in possible_sources:
            if src.is_file() and src.name == name:
                print(f'[数据] 复制 {src} → {target}')
                shutil.copy2(str(src), str(target))
                found = True
                break
            elif src.is_dir() and (src / name).exists():
                print(f'[数据] 复制 {src / name} → {target}')
                shutil.copy2(str(src / name), str(target))
                found = True
                break

        if not found:
            # 搜索压缩包并解压
            for src_dir in possible_sources:
                if src_dir.is_dir():
                    for f in src_dir.glob('*.tar.gz'):
                        print(f'[数据] 解压 {f}')
                        subprocess.run(['tar', '-xzf', str(f), '-C', '.'],
                                     capture_output=True)
                        if target.exists():
                            found = True
                            break
            if not found:
                print(f'[数据] 警告: 未找到 {name}，训练将从 HuggingFace 下载')

    # 验证
    for name in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
        p = cache_dir / name
        if p.exists():
            print(f'[数据] {name}: {p.stat().st_size / 1024 / 1024:.1f} MB')


def train_pcn():
    """PCN 200M 参数训练"""
    cmd = [
        sys.executable, 'train.py',
        '--model', 'pcn', '--no_gating',
        '--d_gate', '128', '--topk', '128',
        '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
        '--lr', '3e-4',
        '--batch_size', '16', '--accum_steps', '2',
        '--seq_len', '256', '--max_seq_len', '256',
        '--max_steps', '10000', '--warmup_steps', '1000',
        '--dataset', 'wikitext',
        '--eval_interval', '1000', '--log_interval', '500',
        '--seed', '0',
        '--exp_name', 'wt_200m_pcn',
    ]
    print(f'[训练] 启动 PCN 200M: {" ".join(cmd)}')
    subprocess.run(cmd)


def train_tf():
    """Transformer 200M 参数训练（对照组）"""
    cmd = [
        sys.executable, 'train.py',
        '--model', 'transformer', '--attn_impl', 'sdpa',
        '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
        '--lr', '1e-3',
        '--batch_size', '16', '--accum_steps', '2',
        '--seq_len', '256', '--max_seq_len', '256',
        '--max_steps', '10000', '--warmup_steps', '1000',
        '--dataset', 'wikitext',
        '--eval_interval', '1000', '--log_interval', '500',
        '--seed', '0',
        '--exp_name', 'wt_200m_tf',
    ]
    print(f'[训练] 启动 Transformer 200M: {" ".join(cmd)}')
    subprocess.run(cmd)


def main():
    print('=' * 60)
    print('PRISM 200M 验证实验（启智社区）')
    print('=' * 60)

    # 环境信息
    print(f'\n[环境] Python: {sys.version}')
    print(f'[环境] 工作目录: {os.getcwd()}')
    print(f'[环境] GPU: {os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()}')

    # 安装依赖
    print('\n--- 安装依赖 ---')
    install_deps()

    # 准备数据
    print('\n--- 准备数据 ---')
    setup_data()

    # 选择训练模式
    mode = 'pcn'
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]

    print(f'\n--- 训练模式: {mode} ---')
    if mode == 'tf':
        train_tf()
    else:
        train_pcn()

    print('\n--- 训练完成 ---')
    # 打印结果
    for exp in ('wt_200m_pcn', 'wt_200m_tf'):
        result_file = Path(f'results/{exp}/results.json')
        if result_file.exists():
            import json
            r = json.loads(result_file.read_text())
            print(f'\n[结果] {exp}:')
            print(f'  Best PPL: {r.get("best_ppl", "N/A")}')
            print(f'  Params: {r.get("n_params", 0) / 1e6:.1f}M')
            print(f'  Val curve: {[(v["step"], round(v["ppl"], 2)) for v in r.get("val_ppl", [])]}')


if __name__ == '__main__':
    main()
