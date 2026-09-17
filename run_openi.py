#!/usr/bin/env python3
"""启智社区训练启动脚本 v3.4——使用 c2net 官方 API

关键改进：
1. 使用 c2net.context.prepare() 获取数据集真实挂载路径
2. 使用 c2net_context.output_path 保存结果（自动回传到启智）
3. 使用项目内 tokenizer/ 目录（无需网络下载）
4. v3.1: 修复 mode 解析——平台实际传参格式与精确匹配 `--mode` 不符时会静默回退
   pcn（导致 TF 任务重跑 PCN）。现兼容 --mode tf / --mode=tf / ----mode tf 等
   形式，默认改为 tf，未知值直接退出；启动时打印 sys.argv 以便日志确诊。
5. v3.2: 新增 tf_scan 模式——TF lr 三点扫描（5e-4 / 2e-3 / 1e-3 复跑）。
   结果：101M 上 TF 最优 ≤5e-4（99.66），63M 扫出的 1e-3 已失效。
6. v3.3: 新增 scan2 模式——对称补扫。终判：调优 TF（99.66）领先 PCN（150.25）
   33.7%；PCN@3e-4 早期全场最佳后失稳崩溃（fp16 激活溢出，权重非 NaN）。
7. v3.4: 新增 v8a 模式——V8 稳定化解锁：① PCN@3e-4 bf16（检验稳定性天花板
   假说：早期优势区间能否被动态范围解锁）② PCN@2e-4 fp16（夹逼中间点）。
   train.py 同步升级：--amp_dtype bf16 + 连续 50 次 NaN-loss 回滚并紧急降 lr。
   每点跑完立即保存，中断不丢已完成部分。
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
                if found.resolve() == target.resolve():
                    print(f'[数据] {name} 已在正确位置')
                    continue
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


def tf_cmd(lr, exp_name):
    """TF 基线统一命令——单点与 lr 扫描共用同一配置，仅 lr/exp_name 不同"""
    return [sys.executable, 'train.py',
            '--model', 'transformer', '--attn_impl', 'sdpa',
            '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
            '--lr', lr, '--batch_size', '16', '--accum_steps', '2',
            '--seq_len', '256', '--max_seq_len', '256',
            '--max_steps', '10000', '--warmup_steps', '2000',
            '--grad_clip', '1.0',
            '--dataset', 'wikitext',
            '--eval_interval', '1000', '--log_interval', '500',
            '--seed', '0', '--exp_name', exp_name]


def pcn_cmd(lr, exp_name, amp_dtype='fp16'):
    """PCN 统一命令——单点与 lr 扫描共用同一配置，仅 lr/exp_name/精度不同"""
    return [sys.executable, 'train.py',
            '--model', 'pcn', '--no_gating',
            '--d_gate', '128', '--topk', '128',
            '--layers', '24', '--d_model', '512', '--ffn_dim', '2048',
            '--lr', lr, '--batch_size', '16', '--accum_steps', '2',
            '--seq_len', '256', '--max_seq_len', '256',
            '--max_steps', '10000', '--warmup_steps', '2000',
            '--grad_clip', '0.5',
            '--dataset', 'wikitext',
            '--eval_interval', '1000', '--log_interval', '500',
            '--seed', '0', '--exp_name', exp_name,
            '--amp_dtype', amp_dtype]


def run_scan(tag, runs):
    """串行跑多点扫描；每点跑完立即保存结果，任务中断不丢已完成部分"""
    codes = []
    for i, (cmd_fn, lr, exp) in enumerate(runs, 1):
        print(f'\n[训练] {tag} {i}/{len(runs)} | lr={lr} | exp={exp}')
        codes.append(subprocess.run(cmd_fn(lr, exp)).returncode)
        save_results()
    return max(codes)


def train(mode='pcn'):
    """启动训练"""
    if mode == 'tf_scan':
        return run_scan('tf_scan', [
            (tf_cmd, '5e-4', 'wt_200m_tf_lr5e-4'),
            (tf_cmd, '2e-3', 'wt_200m_tf_lr2e-3'),
            (tf_cmd, '1e-3', 'wt_200m_tf_lr1e-3')])
    if mode == 'scan2':
        # 对称补扫：TF 探更低点 + PCN 在本规模重扫（此前仅 1e-4）
        return run_scan('scan2', [
            (tf_cmd,  '3e-4', 'wt_200m_tf_lr3e-4'),
            (pcn_cmd, '3e-4', 'wt_090m_pcn_lr3e-4'),
            (pcn_cmd, '5e-5', 'wt_090m_pcn_lr5e-5')])
    if mode == 'v8a':
        # V8 稳定化解锁：① bf16 检验稳定性天花板假说 ② fp16 夹逼中间点
        bf16_pcn = lambda lr, exp: pcn_cmd(lr, exp, amp_dtype='bf16')
        return run_scan('v8a', [
            (bf16_pcn, '3e-4', 'wt_090m_pcn_lr3e-4_bf16'),
            (pcn_cmd,  '2e-4', 'wt_090m_pcn_lr2e-4')])
    if mode == 'tf':
        cmd = tf_cmd('1e-3', 'wt_200m_tf')
    else:
        cmd = pcn_cmd('1e-4', 'wt_090m_pcn_v2')

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
    for exp in ('wt_200m_tf', 'wt_200m_tf_lr5e-4', 'wt_200m_tf_lr1e-3',
                'wt_200m_tf_lr2e-3', 'wt_200m_tf_lr3e-4',
                'wt_090m_pcn_v2', 'wt_090m_pcn_lr3e-4', 'wt_090m_pcn_lr5e-5',
                'wt_090m_pcn_lr3e-4_bf16', 'wt_090m_pcn_lr2e-4'):
        rf = Path(f'results/{exp}/results.json')
        if rf.exists():
            r = json.loads(rf.read_text())
            print(f'\n{"="*50}')
            print(f'结果: {exp}')
            print(f'  Best PPL: {r.get("best_ppl", "N/A")}')
            print(f'  Params: {r.get("n_params", 0)/1e6:.1f}M')
            print(f'  Budget: {r.get("budget", {})}')
            print(f'{"="*50}')


def parse_mode(argv):
    """解析训练模式。默认 tf（仅剩 TF 基线任务），兼容平台各种传参形式。"""
    mode = 'tf'
    for i, arg in enumerate(argv[1:], start=1):
        a = arg.lstrip('-').lower()
        if a == 'mode' and i + 1 < len(argv):
            mode = argv[i + 1].lstrip('-').lower().replace('-', '_')
            break
        if a.startswith('mode='):
            mode = a.split('=', 1)[1].strip().lower().replace('-', '_')
            break
    return mode


def main():
    print('=' * 60)
    print('PRISM 200M 验证实验（启智社区 v3.4）')
    print('=' * 60)
    print(f'[启动参数] sys.argv = {sys.argv}')
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

    mode = parse_mode(sys.argv)
    if mode not in ('pcn', 'tf', 'tf_scan', 'scan2', 'v8a'):
        print(f'!!! 未知 mode: {mode!r}（应为 pcn / tf / tf_scan / scan2 / v8a），退出')
        sys.exit(2)

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
