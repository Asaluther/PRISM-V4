#!/usr/bin/env python3
"""启智社区（OpenI）训练任务启动脚本 v2

修复：
1. 使用本地 tokenizer（不需要 HuggingFace 网络）
2. 搜索更多数据集挂载路径
3. 强制离线模式
"""
import os
import sys

# 强制离线模式（必须在 import transformers 之前设置）
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'

import subprocess
import shutil
from pathlib import Path


def setup_tokenizer():
    """把项目内 tokenizer/ 目录设为 HF 缓存"""
    tok_dir = Path('tokenizer')
    if not tok_dir.exists():
        print('[Tokenizer] 项目内无 tokenizer/ 目录')
        return

    cache_dir = Path('/tmp/hf_cache/models--gpt2/snapshots/local')
    cache_dir.mkdir(parents=True, exist_ok=True)

    for f in tok_dir.glob('*'):
        dest = cache_dir / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))
            print(f'[Tokenizer] 复制 {f.name}')

    # 创建 refs/main 指向 local
    refs_dir = Path('/tmp/hf_cache/models--gpt2/refs')
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / 'main').write_text('local')
    print('[Tokenizer] 本地缓存设置完成')


def find_data_file(name):
    """在所有可能的位置搜索数据文件"""
    search_paths = [
        Path('/dataset'),
        Path('/datasets'),
        Path('/cache/datasets'),
        Path('/data'),
        Path('/mnt/dataset'),
        Path('/tmp/dataset'),
        Path('/tmp/data'),
        Path.cwd(),
        Path.cwd().parent,
    ]

    # 也在 /tmp 下搜索（启智代码目录附近）
    for p in Path('/tmp').glob('**/' + name):
        return p

    # 搜索 tar.gz 并解压
    for root_dir in search_paths:
        if root_dir.is_dir():
            # 直接找文件
            f = root_dir / name
            if f.exists():
                return f
            # 在子目录找
            for sub in root_dir.rglob(name):
                return sub
            # 找 tar.gz 并解压
            for tar in root_dir.rglob('*.tar.gz'):
                print(f'[数据] 解压 {tar}')
                subprocess.run(['tar', '-xzf', str(tar), '-C', str(Path.cwd())],
                             capture_output=True)
                local = Path.cwd() / 'cache' / name
                if local.exists():
                    return local

    return None


def setup_data():
    """准备 WikiText 数据"""
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)

    for name in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
        target = cache_dir / name
        if target.exists():
            print(f'[数据] {name} 已存在 ({target.stat().st_size/1024/1024:.0f} MB)')
            continue

        src = find_data_file(name)
        if src:
            print(f'[数据] 找到 {src}，复制到 {target}')
            shutil.copy2(str(src), str(target))
            print(f'[数据] {name}: {target.stat().st_size/1024/1024:.0f} MB')
        else:
            print(f'[数据] 警告: 未找到 {name}')

    # 列出搜索过的目录内容（调试用）
    print('[数据] 搜索路径内容:')
    for d in ['/dataset', '/datasets', '/cache', '/data', '/tmp']:
        p = Path(d)
        if p.exists():
            items = list(p.iterdir())[:10]
            print(f'  {d}: {[i.name for i in items]}')


def train_pcn():
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
    print(f'[训练] 启动 PCN 200M')
    result = subprocess.run(cmd)
    return result.returncode


def train_tf():
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
    print(f'[训练] 启动 Transformer 200M')
    result = subprocess.run(cmd)
    return result.returncode


def main():
    print('=' * 60)
    print('PRISM 200M 验证实验（启智社区 v2）')
    print('=' * 60)

    print(f'\n[环境] Python: {sys.version}')
    print(f'[环境] 工作目录: {os.getcwd()}')
    os.system('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')

    # 列出 /tmp 下的内容（调试）
    print('\n[调试] /tmp 下的目录:')
    for d in sorted(Path('/tmp').iterdir()):
        if d.is_dir():
            try:
                items = list(d.iterdir())[:5]
                print(f'  {d}: {[i.name for i in items]}')
            except:
                pass

    print('\n--- 设置 Tokenizer（本地模式）---')
    setup_tokenizer()

    print('\n--- 准备数据 ---')
    setup_data()

    # 检查关键文件是否存在
    train_py = Path('train.py')
    print(f'\n[检查] train.py 存在: {train_py.exists()}')
    print(f'[检查] src/models/pcn.py 存在: {Path("src/models/pcn.py").exists()}')
    print(f'[检查] cache/wikitext_train_int32.npy 存在: {Path("cache/wikitext_train_int32.npy").exists()}')

    mode = 'pcn'
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]

    print(f'\n--- 训练模式: {mode} ---')
    if mode == 'tf':
        code = train_tf()
    else:
        code = train_pcn()

    print(f'\n--- 训练退出码: {code} ---')

    if code != 0:
        print('!!! 训练失败 !!!')
        sys.exit(1)

    import json
    for exp in ('wt_200m_pcn', 'wt_200m_tf'):
        result_file = Path(f'results/{exp}/results.json')
        if result_file.exists():
            r = json.loads(result_file.read_text())
            print(f'\n[结果] {exp}:')
            print(f'  Best PPL: {r.get("best_ppl", "N/A")}')
            print(f'  Params: {r.get("n_params", 0) / 1e6:.1f}M')
            print(f'  Budget: {r.get("budget", {})}')


if __name__ == '__main__':
    main()
