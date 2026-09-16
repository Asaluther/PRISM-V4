"""PRISM V5 — WikiText-103 数据加载（真实域）

与 tinystories.py 同协议：tokenize 全量 → 等长 chunk（seq_len+1）→ (input, target)。
内存：103M tokens 以 numpy int32 存储（~412MB）；分块 tokenize 中间态设 20M 落盘阈值。

首次使用需在线下载（shell 层覆盖离线变量）：
  HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 \
  python -c "from src.data.wikitext import WikiTextDataset; WikiTextDataset('train')"
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset

os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

CONFIG = 'wikitext-103-raw-v1'
FLUSH_EVERY = 20_000_000  # 分块 tokenize 的中间 list 落盘阈值（tokens）
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'cache'


class WikiTextDataset(Dataset):
    def __init__(self, split='train', seq_len=256, max_tokens=None):
        self.seq_len = seq_len
        # 优先从项目内 tokenizer/ 目录加载（离线环境），否则从 HuggingFace 下载
        _tok_dir = Path(__file__).resolve().parent.parent.parent / 'tokenizer'
        if (_tok_dir / 'vocab.json').exists():
            self.tokenizer = GPT2TokenizerFast.from_pretrained(str(_tok_dir))
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f'wikitext_{split}_int32.npy'
        if cache_file.exists():
            all_tokens = np.load(cache_file, mmap_mode='r')
            print(f"  WikiText {split}: 从缓存加载 {len(all_tokens)/1e6:.1f}M tokens")
        else:
            ds = load_dataset('Salesforce/wikitext', CONFIG, split=split)
            texts = ds['text']
            chunks = []
            buf = []
            total = 0
            B = 2000
            for i in range(0, len(texts), B):
                enc = self.tokenizer(texts[i:i + B], add_special_tokens=False)['input_ids']
                for ids in enc:
                    buf.extend(ids)
                if len(buf) >= FLUSH_EVERY:
                    chunks.append(np.array(buf, dtype=np.int32))
                    total += len(buf)
                    buf = []
                    print(f"  tokenized ~{total/1e6:.0f}M tokens...")
                if max_tokens and total + len(buf) >= max_tokens:
                    break
            if buf:
                chunks.append(np.array(buf, dtype=np.int32))
            all_tokens = np.concatenate(chunks)
            np.save(cache_file, np.asarray(all_tokens, dtype=np.int32))
            print(f"  WikiText {split}: tokenize 完成 {len(all_tokens)/1e6:.1f}M tokens，已缓存至 {cache_file.name}")

        chunk_size = seq_len + 1
        n_chunks = len(all_tokens) // chunk_size
        self.data = np.asarray(all_tokens[:n_chunks * chunk_size]).reshape(n_chunks, chunk_size)
        print(f"  WikiText {split}: -> {n_chunks} chunks @ seq_len={seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        return chunk[:-1], chunk[1:]  # input, target


def get_wikitext_loaders(seq_len=256, batch_size=32, num_workers=4,
                         max_train=None, max_val=None):
    train_ds = WikiTextDataset(split='train', seq_len=seq_len, max_tokens=max_train)
    val_ds = WikiTextDataset(split='validation', seq_len=seq_len, max_tokens=max_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.tokenizer
