"""
PRISM V4 — Transformer 对照基线

标准 GPT-2 风格，参数量与 PCN 对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """标准 Transformer block (Pre-LN, causal)

    attn_impl:
      'sdpa' — F.scaled_dot_product_attention(is_causal=True)，标准快路径（默认）
      'mha'  — nn.MultiheadAttention + bool mask，兼容旧 checkpoint（2026-09-08 前）
    """

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1, attn_impl='sdpa'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_impl = attn_impl
        if attn_impl == 'mha':
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        elif attn_impl == 'decay':
            # 第三架构代理（RetNet/GLA 线性注意力家族的最小形式）：
            # softmax(QK^T + lambda*(t-i)) 因果——每头可学习衰减率 lambda。
            # SSM「固定衰减核」精神的桥接；与 attention 同构无实现争议。
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.decay_log = nn.Parameter(torch.full((n_heads,), -2.0))
            self.attn_drop = nn.Dropout(dropout)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.attn_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, D = x.shape
        h = self.ln1(x)
        if self.attn_impl == 'mha':
            mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
            attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        else:
            qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, N, dh]
            dt = torch.arange(N, device=x.device).unsqueeze(1) - torch.arange(N, device=x.device).unsqueeze(0)  # dt[q,k] = q - k
            causal = dt >= 0  # key 在过去（k <= q）
            if self.attn_impl == 'decay':
                # [H, N, N] 浮点偏置：因果内 lambda*(q-k)，因果外 -inf
                bias = self.decay_log.view(-1, 1, 1) * dt.unsqueeze(0)
                bias = bias.masked_fill(~causal.unsqueeze(0), float('-inf'))
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = self.proj(o.transpose(1, 2).reshape(B, N, D))
            attn_out = self.attn_drop(attn_out)
        x = x + attn_out

        # FFN
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


class TransformerModel(nn.Module):
    """标准 Transformer LM — 对照基线"""

    def __init__(self, vocab_size, d_model=256, n_layers=12, n_heads=4,
                 ffn_dim=1024, dropout=0.1, max_seq_len=512, attn_impl='sdpa'):
        super().__init__()

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout, attn_impl=attn_impl)
            for _ in range(n_layers)
        ])

        # 输出
        self.ln_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, return_stats=False):
        B, N = input_ids.shape

        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(N, device=input_ids.device))
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_out(x)
        logits = self.lm_head(x)

        if return_stats:
            return logits, {}
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
