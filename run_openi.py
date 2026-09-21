#!/usr/bin/env python3
"""启智社区训练启动脚本 v3.5——使用 c2net 官方 API

v3.5 新增 m500 系列（500M 判决实验，W1/B4 后续）：
  m500_hyb      HYB 12+12 d1280/L24 骨干5e-4/头1e-4 s0（~516M，~4.8h V100S）
  m500_hyb1e3   HYB 骨干 1e-3 夹逼臂（330M 最优上漂，500M 必须上探）
  m500_tf       TF 24L d1280 @5e-4 s0（~553M，~5.8h）
  m500_hyb_s1 / m500_tf_s1  第二 seed（配对误差条，积分富余时）
  TF 向下夹逼（2.5e-4）省略：101M/330M 双点已证 5e-4 近优且 1e-3 发散。
  协议与 330M 点（run_d1024_reinforce.sh）逐字段一致，仅 d_model/n_heads/
  ffn_dim 升档。VRAM 估 19-22GB（V100S-32GB 可容，4080 16GB 不可）。

关键改进（历史）：
1. 使用 c2net.context.prepare() 获取数据集真实挂载路径
2. 使用 c2net_context.output_path 保存结果（自动回传到启智）
3. 使用项目内 tokenizer/ 目录（无需网络下载）
4. v3.1: 修复 mode 解析；v3.2: tf_scan；v3.3: scan2 对称补扫；
   v3.4: v8a 稳定化解锁。每点跑完立即保存，中断不丢已完成部分。
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


def m500_hyb_cmd(backbone_lr, exp_name, seed=0):
    """500M 判决实验——HYB 12+12 d1280/L24（与 330M 协议逐字段一致，仅升档）"""
    return [sys.executable, 'train.py',
            '--model', 'hybrid', '--no_gating',
            '--tf_layers', '12',
            '--d_gate', '128', '--topk', '128',
            '--layers', '24', '--d_model', '1280', '--n_heads', '10',
            '--ffn_dim', '5120',
            '--lr', '1e-4', '--lr_backbone', backbone_lr,
            '--batch_size', '16', '--accum_steps', '2',
            '--seq_len', '256', '--max_seq_len', '256',
            '--max_steps', '20000', '--warmup_steps', '4000',
            '--grad_clip', '0.5',
            '--dataset', 'wikitext',
            '--eval_interval', '1000', '--log_interval', '500',
            '--seed', str(seed), '--exp_name', exp_name,
            '--amp_dtype', 'fp16']


def m500_tf_cmd(exp_name, seed=0):
    """500M 判决实验——TF 24L d1280 @5e-4（330M 双侧夹逼直承值）"""
    return [sys.executable, 'train.py',
            '--model', 'transformer', '--attn_impl', 'sdpa',
            '--layers', '24', '--d_model', '1280', '--n_heads', '10',
            '--ffn_dim', '5120',
            '--lr', '5e-4',
            '--batch_size', '16', '--accum_steps', '2',
            '--seq_len', '256', '--max_seq_len', '256',
            '--max_steps', '20000', '--warmup_steps', '4000',
            '--grad_clip', '0.5',
            '--dataset', 'wikitext',
            '--eval_interval', '1000', '--log_interval', '500',
            '--seed', str(seed), '--exp_name', exp_name,
            '--amp_dtype', 'fp16']


def m500_tf_cmd_25e4(seed=0):
    """500M TF @2.5e-4 夹逼臂——堵「TF 中低 lr 窗口」反事实"""
    cmd = m500_tf_cmd(f'wt_553m_tf_lr2.5e-4_s0', seed)
    # 替换 lr
    idx = cmd.index('--lr')
    cmd[idx + 1] = '2.5e-4'
    return cmd


def train(mode='pcn'):
    """启动训练"""
    m500 = {
        'm500_hyb':     lambda: m500_hyb_cmd('5e-4', 'wt_516m_hyb_ng_lr5e-4_s0'),
        'm500_hyb1e3':  lambda: m500_hyb_cmd('1e-3', 'wt_516m_hyb_ng_lr1e-3_s0'),
        'm500_tf':      lambda: m500_tf_cmd('wt_553m_tf_lr5e-4_s0'),
        'm500_tf25e4':  lambda: m500_tf_cmd_25e4(),
        'm500_hyb_s1':  lambda: m500_hyb_cmd('5e-4', 'wt_516m_hyb_ng_lr5e-4_s1', 1),
        'm500_tf_s1':   lambda: m500_tf_cmd('wt_553m_tf_lr5e-4_s1', 1),
    }
    if mode in m500:
        print(f'[训练] 500M 判决: {mode}')
        result = subprocess.run(m500[mode]())
        return result.returncode
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

        # 复制检查点（500M 教训：容器销毁即丢失；c2net 支持大文件回传）
        for ckpt_file in results_dir.glob('**/best_model.pt'):
            exp_name = ckpt_file.parent.name
            dest = output_path / exp_name / 'best_model.pt'
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ckpt_file), str(dest))
                print(f'[输出] 已保存 {exp_name}/best_model.pt '
                      f'({ckpt_file.stat().st_size/1024/1024:.0f} MB)')

        # 也复制训练日志
        for log_file in results_dir.glob('**/train.log'):
            exp_name = log_file.parent.name
            dest = output_path / exp_name / 'train.log'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(log_file), str(dest))

    # 打印结果摘要
    import json
    for exp in ('wt_516m_hyb_ng_lr5e-4_s0', 'wt_516m_hyb_ng_lr1e-3_s0',
                'wt_553m_tf_lr5e-4_s0', 'wt_516m_hyb_ng_lr5e-4_s1',
                'wt_553m_tf_lr5e-4_s1',
                'wt_200m_tf', 'wt_200m_tf_lr5e-4', 'wt_200m_tf_lr1e-3',
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
    if mode not in ('pcn', 'tf', 'tf_scan', 'scan2', 'v8a',
                    'm500_hyb', 'm500_hyb1e3', 'm500_tf', 'm500_tf25e4',
                    'm500_hyb_s1', 'm500_tf_s1'):
        print(f'!!! 未知 mode: {mode!r}，退出')
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
