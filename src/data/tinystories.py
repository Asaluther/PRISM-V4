"""PRISM V4 — TinyStories 数据加载"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# 使用 hf-mirror.com 避免 DNS 污染导致 huggingface.co 不可达
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 优先使用本地缓存，避免网络不可达时崩溃
# （首次下载数据时需显式覆盖为 0，见 run 脚本）
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


class TinyStoriesDataset(Dataset):
    def __init__(self, split='train', tokenizer_name='gpt2', seq_len=256, max_examples=None):
        self.seq_len = seq_len
        _tok_dir = Path(__file__).resolve().parent.parent.parent / 'tokenizer'
        if (_tok_dir / 'vocab.json').exists():
            self.tokenizer = GPT2TokenizerFast.from_pretrained(str(_tok_dir))
        else:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ds = load_dataset('roneneldan/TinyStories', split=split)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        # tokenize and flatten
        all_tokens = []
        for example in ds:
            tokens = self.tokenizer.encode(example['text'])
            all_tokens.extend(tokens)
            if len(all_tokens) >= max_examples * seq_len if max_examples else float('inf'):
                break

        # 等分成 seq_len+1 长度的块
        total_len = len(all_tokens)
        self.data = []
        chunk_size = seq_len + 1
        for i in range(0, total_len - chunk_size + 1, chunk_size):
            chunk = all_tokens[i:i + chunk_size]
            if len(chunk) == chunk_size:
                self.data.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # input, target


def get_dataloaders(seq_len=256, batch_size=32, num_workers=4, max_train=50000, max_val=5000):
    train_ds = TinyStoriesDataset(split='train', seq_len=seq_len, max_examples=max_train)
    val_ds = TinyStoriesDataset(split='validation', seq_len=seq_len, max_examples=max_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.tokenizer
