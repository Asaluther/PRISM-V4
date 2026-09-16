#!/usr/bin/env python3
"""启智社区训练启动脚本 v3——使用 c2net 官方 API

关键改进：
1. 使用 c2net.context.prepare() 获取数据集真实挂载路径
2. 使用 c2net_context.output_path 保存结果（自动回传到启智）
3. 使用项目内 tokenizer/ 目录（无需网络下载）
"""
import os
import sys

# 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# c2net 必须在最前面导入
try:
    from c2net.context import prepare, upload_output
    c2net_context = prepare()
    print('[c2net] 数据集路径:', c2net_context.dataset_path)
    print('[c2net] 输出路径:', c2net_context.output_path)
    HAS_C2NET = True
except ImportError:
    print('[c2net] 未安装（可能不在启智环境）')
    HAS_C2NET = False

import subprocess
import shutil
from pathlib import Path


def setup_tokenizer():
    """把项目内 tokenizer/ 设为 HF 本地缓存"""
    tok_dir = Path(__file__).parent / 'tokenizer'
    if not tok_dir.exists():
        print('[Tokenizer] 无本地 tokenizer 目录')
        return

    # 创建 HF 缓存结构
    cache_base = Path(os.environ.get('TRANSFORMERS_CACHE', '/tmp/hf_cache'))
    model_dir = cache_base / 'models--gpt2' / 'snapshots' / 'local'
    refs_dir = cache_base / 'models--gpt2' / 'refs'
    model_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    for f in tok_dir.glob('*'):
        shutil.copy2(str(f), str(model_dir / f.name))
    (refs_dir / 'main').write_text('local')

    # 也设置 HF_HOME 确保生效
    os.environ['HF_HOME'] = str(cache_base)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_base / 'models--gpt2')

    print(f'[Tokenizer] 本地缓存 → {model_dir}')


def setup_data():
    """用 c2net 路径准备数据"""
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)

    if HAS_C2NET:
        ds_path = Path(c2net_context.dataset_path)
        print(f'[数据] c2net 数据集目录: {ds_path}')
        print(f'[数据] 目录内容: {[d.name for d in ds_path.iterdir()]}')

        # 遍历数据集目录找 .npy 文件
        for name in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
            target = cache_dir / name
            if target.exists():
                print(f'[数据] {name} 已存在')
                continue

            # 在数据集目录中搜索
            for found in ds_path.rglob(name):
                print(f'[数据] 找到 {found}，复制...')
                shutil.copy2(str(found), str(target))
                print(f'[数据] {name}: {target.stat().st_size/1024/1024:.0f} MB')
                break

            # 也搜索 tar.gz
            if not target.exists():
                for tar in ds_path.rglob('*.tar.gz'):
                    print(f'[数据] 解压 {tar}')
                    subprocess.run(['tar', '-xzf', str(tar), '-C', str(cache_dir.parent)],
                                 capture_output=True)
                    if target.exists():
                        print(f'[数据] {name}: {target.stat().st_size/1024/1024:.0f} MB')
                        break

    # 验证
    for name in ('wikitext_train_int32.npy', 'wikitext_validation_int32.npy'):
        p = cache_dir / name
        status = f'{p.stat().st_size/1024/1024:.0f} MB' if p.exists() else 'MISSING!'
        print(f'[数据] {name}: {status}')


def train(mode='pcn'):
    """启动训练"""
    if mode == 'tf':
        cmd = [sys.executable, 'train.py',
               '--model', 'transformer', '--attn_impl', 'sdpa',
               '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
               '--lr', '1e-3', '--batch_size', '16', '--accum_steps', '2',
               '--seq_len', '256', '--max_seq_len', '256',
               '--max_steps', '10000', '--warmup_steps', '1000',
               '--dataset', 'wikitext',
               '--eval_interval', '1000', '--log_interval', '500',
               '--seed', '0', '--exp_name', 'wt_200m_tf']
    else:
        cmd = [sys.executable, 'train.py',
               '--model', 'pcn', '--no_gating',
               '--d_gate', '128', '--topk', '128',
               '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
               '--lr', '3e-4', '--batch_size', '16', '--accum_steps', '2',
               '--seq_len', '256', '--max_seq_len', '256',
               '--max_steps', '10000', '--warmup_steps', '1000',
               '--dataset', 'wikitext',
               '--eval_interval', '1000', '--log_interval', '500',
               '--seed', '0', '--exp_name', 'wt_200m_pcn']

    print(f'[训练] 启动 {mode} 200M')
    result = subprocess.run(cmd)
    return result.returncode


def save_results():
    """把结果复制到 c2net 输出路径（自动回传启智）"""
    if not HAS_C2NET:
        return

    output_path = Path(c2net_context.output_path)
    results_dir = Path('results')

    if results_dir.exists():
        for json_file in results_dir.glob('**/results.json'):
            exp_name = json_file.parent.name
            dest = output_path / exp_name / 'results.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(json_file), str(dest))
            print(f'[输出] 已保存 {exp_name}/results.json → {dest}')

        # 也复制训练日志
        for log_file in results_dir.glob('**/train.log'):
            exp_name = log_file.parent.name
            dest = output_path / exp_name / 'train.log'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(log_file), str(dest))

    # 打印结果摘要
    import json
    for exp in ('wt_200m_pcn', 'wt_200m_tf'):
        rf = Path(f'results/{exp}/results.json')
        if rf.exists():
            r = json.loads(rf.read_text())
            print(f'\n{"="*50}')
            print(f'结果: {exp}')
            print(f'  Best PPL: {r.get("best_ppl", "N/A")}')
            print(f'  Params: {r.get("n_params", 0)/1e6:.1f}M')
            print(f'  Budget: {r.get("budget", {})}')
            print(f'{"="*50}')


def main():
    print('=' * 60)
    print('PRISM 200M 验证实验（启智社区 v3）')
    print('=' * 60)
    print(f'\n[环境] Python: {sys.version}')
    print(f'[环境] 工作目录: {os.getcwd()}')
    os.system('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')

    print('\n--- Tokenizer ---')
    setup_tokenizer()

    print('\n--- 数据 ---')
    setup_data()

    # 确认关键文件
    print(f'\n[检查] train.py: {Path("train.py").exists()}')
    print(f'[检查] src/models/pcn.py: {Path("src/models/pcn.py").exists()}')
    print(f'[检查] data: {Path("cache/wikitext_train_int32.npy").exists()}')

    mode = 'pcn'
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]

    print(f'\n--- 开始训练: {mode} ---')
    code = train(mode)

    print(f'\n--- 训练退出码: {code} ---')

    # 保存结果
    save_results()

    # 回传到启智
    if HAS_C2NET:
        print('\n[c2net] 回传结果...')
        upload_output()

    if code != 0:
        print('!!! 训练失败 !!!')
        sys.exit(1)
    print('\n✅ 全部完成')


if __name__ == '__main__':
    main()
