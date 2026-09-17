"""
PRISM V4 — 混合架构：TF 骨干 + PCN 误差流适应头

依据 90M 双判决（results/SCALE_090M_VALIDATION.md）：TF 赢预训练（数据充裕区），
PCN 赢全部适应协议（数据稀缺区）。混合架构让两个组件各自站在被验证的优势区：
底部 TransformerBlock 提供高质量通用表征，顶部 PCNBlock（no_gating 误差流路径）
提供终端边用边学能力。

接口依据：PCNLayer 对输入无 embedding 假设——W_bu(LN_bu(x)) 先归一化再提取、
误差 ±10 硬截断、W_res 恒等残差与 Pre-LN 残差块输出契约对齐，TF 输出可直接接入。

命名约定：self.layers = TF 块列表。冷启动/流式协议的「冻结底部 12 层」
（model.layers[:12]）对 hybrid 恰好冻结全部 TF 骨干——适应只发生在 PCN 头，
协议可比性由构造保证，无需改协议代码。
"""

import torch
import torch.nn as nn

from src.models.transformer import TransformerBlock
from src.models.pcn import PCNBlock


class HybridModel(nn.Module):
    """TF 骨干（底部）+ PCN 误差流头（顶部）+ 共享 embedding/输出头（绑定）"""

    def __init__(self, vocab_size, d_model=512, n_tf_layers=12, n_pcn_layers=12,
                 n_heads=4, ffn_dim=2048, d_gate=128, dropout=0.1,
                 max_seq_len=256, attn_impl='sdpa'):
        super().__init__()
        self.d_model = d_model
        self.n_tf_layers = n_tf_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout, attn_impl=attn_impl)
            for _ in range(n_tf_layers)
        ])
        self.pcn_blocks = nn.ModuleList([
            PCNBlock(d_model, n_heads=n_heads, d_gate=d_gate, dropout=dropout,
                     gate_ln=True, use_ffn=False, use_bmm_gate=True)
            for _ in range(n_pcn_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # 通用初始化与双亲一致（TransformerModel/PCNModel 均 std=0.02）
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # PCN 专用初始化（对齐 PCNModel._init_weights, init_mode='fixed'）：
        # W_td 小值渐进引入反馈；W_res 恒等恢复信号/梯度直通；w_g Xavier 解门控死锁
        for blk in self.pcn_blocks:
            pc = blk.pcn_layer
            nn.init.normal_(pc.W_td.weight, mean=0.0, std=0.01)
            nn.init.eye_(pc.W_res.weight)
            nn.init.xavier_uniform_(pc.w_g.weight)

    def forward(self, input_ids, topk=64, no_feedback=False, no_gating=True,
                return_stats=False, feedback_mode='prev_layer', n_pass=1):
        """PCNModel 前向的混合版：embedding → TF 循环 → PCN 循环 → 输出头。
        MVP 仅支持 prev_layer / n_pass=1（与 90M 判决实验同配置）；
        no_gating 默认 True——误差流头是混合架构的存在理由。"""
        assert feedback_mode == 'prev_layer' and n_pass == 1, \
            'HybridModel MVP 仅支持 feedback_mode=prev_layer / n_pass=1'
        B, N = input_ids.shape

        x = self.token_emb(input_ids) + \
            self.pos_emb(torch.arange(N, device=input_ids.device))
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x)

        # PCN 头：与 PCNModel 默认分支（逐层串联，h_above=上一层输出）同构；
        # 首个 PCN 层 h_above=None（纯提取，x=TF 骨干输出）
        prev_h = None
        for i, blk in enumerate(self.pcn_blocks):
            h_above = prev_h if (i > 0 and not no_feedback) else None
            x = blk(x, h_above=h_above, topk=topk,
                    no_feedback=no_feedback, no_gating=no_gating)
            prev_h = x

        x = self.ln_out(x)
        logits = self.lm_head(x)

        if return_stats:
            return logits, []
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
