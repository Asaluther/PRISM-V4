"""
PRISM V4 — PCN (Predictive Coding Network)

核心计算基元：预测误差驱动的层级双向信息流
- bottom-up: 特征提取
- top-down: 层级预测
- 预测误差: e = u - p
- 跨位置交互: 误差驱动 Top-K 门控（非 attention）
- 状态更新: 误差修正 + 残差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PCNLayer(nn.Module):
    """预测编码层 — 双向信息流 + 误差驱动门控"""

    def __init__(self, d_model, d_gate=32, dropout=0.1, gate_ln=True, gate_softmax=False,
                 gate_source='feat', val_source='wlat_e', use_bmm_gate=True):
        super().__init__()
        self.d = d_model
        # 门控输入归一化（无参数，不改变 checkpoint 结构）。
        # 诊断 step1：无归一化时 e_comp 量级 ~1e-2，g ≈ sigmoid(0) ≈ 0.5 恒定，
        # w_g 梯度 ~1e-8 → 门控死锁。归一化后 g 初始有方差，w_g 可学习。
        self.gate_ln = gate_ln
        # 机理实验（R2c）：True 时门控用 softmax 竞争归一化替代独立 sigmoid，
        # 用于检验「独立门 vs 竞争归一化」是否是复制任务能力差异的机制根源
        self.gate_softmax = gate_softmax
        # P3 门来源/值通路 2×2 分解开关（默认 = 原生行为）
        # gate_source: 'feat' 压缩特征交互门（原生） | 'qk' q·k 相似度门
        # val_source:  'wlat_e' W_lat(e) 误差变换值（原生） | 'wlat_h' W_lat(norm(x)) 输入变换值
        self.gate_source = gate_source
        self.val_source = val_source
        # T1a 等价加速开关（bmm+偏置形式，验证后默认开）
        self.use_bmm_gate = use_bmm_gate
        if gate_source == 'qk':
            self.W_q = nn.Linear(d_model, d_gate, bias=False)
            self.W_k = nn.Linear(d_model, d_gate, bias=False)

        # 步骤1: 自底向上特征提取
        self.norm_bu = nn.LayerNorm(d_model)
        self.W_bu = nn.Linear(d_model, d_model, bias=False)

        # 步骤2: 自顶向下预测
        self.W_td = nn.Linear(d_model, d_model, bias=False)

        # 步骤4: 跨位置交互 — 先压缩再交互（内存优化）
        self.W_compress = nn.Linear(d_model, d_gate, bias=False)
        self.W_lat = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(2 * d_gate, 1, bias=False)

        # 步骤5: 状态更新
        self.norm_update = nn.LayerNorm(d_model)
        self.W_up = nn.Linear(d_model, d_model, bias=False)
        self.W_res = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h_above=None, topk=64, return_stats=False):
        """
        x:        [B, N, D] — 来自下层的输入（首层为 embedding）
        h_above:  [B, N, D] — 来自上层的状态（首层为 None/零）
        topk:      每个位置最多关注的邻居数
        return_stats: 是否返回误差/门控统计（分析用）
        """
        B, N, D = x.shape

        # === 步骤1: 自底向上特征 ===
        u = self.norm_bu(x)
        u = self.W_bu(u)

        # === 步骤2: 自顶向下预测 ===
        if h_above is not None:
            p = self.W_td(h_above)
        else:
            p = torch.zeros_like(u)

        # === 步骤3: 预测误差 ===
        e = u - p  # [B, N, D]
        e = torch.clamp(e, min=-10.0, max=10.0)  # 防止梯度爆炸

        # 轻量记录（局部学习规则实验用，detach 无梯度开销）
        self.last_e = e.detach()
        self.last_h_above = h_above.detach() if h_above is not None else None
        self.last_x = x.detach()

        # === 步骤4: 误差驱动的跨位置交互（Top-K） ===
        if self.gate_source == 'qk':
            # P3 分解：门来自 q·k 相似度（层输入的压缩投影）
            h_n = self.norm_bu(x)
            q = self.W_q(h_n)  # [B, N, d_gate]
            k = self.W_k(h_n)
            if self.gate_ln:
                q = F.layer_norm(q, (q.size(-1),))
                k = F.layer_norm(k, (k.size(-1),))
            gate = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))  # [B, N, N]
            gate = torch.sigmoid(gate) if not self.gate_softmax else gate
        else:
            # 原生：压缩特征交互门。
            # 两条等价路径（T1a 已验证 max diff 1.2e-07）：
            #  - use_bmm_gate=True：logit_ij = q_i·k_j + b_i + c_j（bmm+偏置，
            #    消掉 chunk 循环与 [B,c,N,2dg] 交互张量，快 2-4x）
            #  - False：原 chunk 循环（保留作对照/校准格）
            e_comp = self.W_compress(e)  # [B, N, d_gate]
            if self.gate_ln:
                e_comp = F.layer_norm(e_comp, (e_comp.size(-1),))

            if getattr(self, 'use_bmm_gate', False):
                w_g = self.w_g.weight  # [1, 2*dg]
                dg = e_comp.size(-1)
                w_a, w_b = w_g[0, :dg], w_g[0, dg:]
                q = e_comp * w_a                       # [B,N,dg]
                logits = q @ e_comp.transpose(-1, -2)  # [B,N,N]
                eb = e_comp @ w_b                      # [B,N]
                logits = logits + eb.unsqueeze(-1) + eb.unsqueeze(-2)
                gate = torch.sigmoid(logits)
            else:
                gate_values = []
                chunk_size = min(N, 128)
                for i_start in range(0, N, chunk_size):
                    i_end = min(i_start + chunk_size, N)
                    e_i = e_comp[:, i_start:i_end, :].unsqueeze(2)
                    e_j = e_comp.unsqueeze(1)
                    interaction = torch.cat([e_i * e_j, e_i + e_j], dim=-1)
                    if self.gate_softmax:
                        g_chunk = self.w_g(interaction).squeeze(-1)
                    else:
                        g_chunk = torch.sigmoid(self.w_g(interaction).squeeze(-1))
                    gate_values.append(g_chunk)
                gate = torch.cat(gate_values, dim=1)  # [B, N, N]

        # 因果约束（2026-09-08 修复）：位置 i 只能聚合 j <= i。
        # 此前无掩码，门控可直接读取未来 token，"预测下一词"退化为抄写。
        causal = torch.tril(torch.ones(N, N, device=gate.device, dtype=torch.bool))
        if self.gate_softmax:
            gate = gate.masked_fill(~causal, float('-inf'))
            gate = torch.softmax(gate, dim=-1)
            gate = torch.nan_to_num(gate, nan=0.0)  # 全 -inf 行（不可能出现，保底）
            gate_sparse = gate  # softmax 版不做 Top-K（保持归一化语义）
        else:
            gate = gate.masked_fill(~causal, 0.0)
            # Top-K 稀疏化（在因果范围内；前 K-1 个位置可选邻居不足 K 时，
            # 多余的 topk 值为 0，scatter 后零贡献，无副作用）
            if topk < N:
                topk_vals, topk_idx = gate.topk(topk, dim=-1)
                gate_sparse = torch.zeros_like(gate).scatter_(-1, topk_idx, topk_vals)
            else:
                gate_sparse = gate

        # 加权聚合（P3 值通路分解：误差变换值 vs 输入变换值）
        val = e if self.val_source == 'wlat_e' else self.norm_bu(x)
        lat = torch.matmul(gate_sparse, self.W_lat(val))  # [B, N, D]

        # === 步骤5: 状态更新 ===
        update = self.W_up(self.norm_update(e + lat))
        h = F.gelu(update) + self.W_res(x)
        h = self.dropout(h)

        if return_stats:
            with torch.no_grad():
                error_norms = e.norm(dim=-1)  # [B, N]
                gate_entropy = -(gate_sparse * (gate_sparse + 1e-10).log()).sum(-1)  # [B, N]
                stats = {
                    'error_norms': error_norms,
                    'gate_entropy': gate_entropy,
                    'gate_mean': gate_sparse.mean(dim=(0, -1)),  # [N]
                }
            return h, stats
        return h


class PCNBlock(nn.Module):
    """带可选 no_attention 回退的 PCN 层（消融用）"""

    def __init__(self, d_model, n_heads=4, d_gate=32, dropout=0.1, gate_ln=True,
                 gate_softmax=False, gate_source='feat', val_source='wlat_e',
                 use_ffn=False, use_bmm_gate=True):
        super().__init__()
        self.pcn_layer = PCNLayer(d_model, d_gate=d_gate, dropout=dropout,
                                  gate_ln=gate_ln, gate_softmax=gate_softmax,
                                  gate_source=gate_source, val_source=val_source,
                                  use_bmm_gate=use_bmm_gate)
        # 机制二分（块子件）：True 时块尾加标准 FFN 子层——
        # 假说：FFN 的逐位置通道混合稀释位置 token 身份，干扰精确复制
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn_ln = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        # 消融: no_gating 时用标准 attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, h_above=None, topk=64, no_feedback=False, no_gating=False, return_stats=False):
        if no_gating:
            # 消融（V4_PLAN 实验 2 定义）：仅将误差门控聚合换成标准 attention。
            # 误差流 e = u - p、反馈、状态更新、残差全部保留——
            # 三臂对照各自隔离一个机制：full / no_feedback(去反馈) / no_gating(去门控)
            pc = self.pcn_layer
            u = pc.W_bu(pc.norm_bu(x))
            if h_above is not None:
                p = pc.W_td(h_above)
            else:
                p = torch.zeros_like(u)
            e = (u - p).clamp(-10.0, 10.0)
            # 局部学习规则记录（与 PCNLayer.forward 同约定）
            pc.last_e = e.detach()
            pc.last_h_above = h_above.detach() if h_above is not None else None
            pc.last_x = x.detach()
            # 因果掩码与主路径一致（no_gating 消融同样不允许看未来）
            mask = torch.triu(torch.ones(x.size(1), x.size(1),
                                         device=x.device, dtype=torch.bool), diagonal=1)
            # need_weights=False：权重矩阵全仓无读取方，True 会强制 MHA 慢路径
            # （b1 终端口径 -13% 吞吐，T5 优化项）
            attn_out, _ = self.attn(e, e, e, attn_mask=mask, need_weights=False)
            self.last_attn_w = None
            lat = self.attn_out(attn_out)
            update = pc.W_up(pc.norm_update(e + lat))
            h = F.gelu(update) + pc.W_res(x)
            if self.use_ffn:
                h = h + self.ffn(self.ffn_ln(h))
            h = pc.dropout(h)
            return h
        h = self.pcn_layer(x, h_above=h_above, topk=topk, return_stats=return_stats)
        if self.use_ffn:
            h = h + self.ffn(self.ffn_ln(h))
        return h


class PCNModel(nn.Module):
    """预测编码网络 — 完整模型"""

    def __init__(self, vocab_size, d_model=256, n_layers=12, n_heads=4,
                 d_gate=32, dropout=0.1, max_seq_len=512, init_mode='fixed',
                 gate_softmax=False, gate_source='feat', val_source='wlat_e',
                 use_ffn=False, use_bmm_gate=True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.init_mode = init_mode

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # PCN 层
        self.layers = nn.ModuleList([
            PCNBlock(d_model, n_heads=n_heads, d_gate=d_gate, dropout=dropout,
                     gate_ln=(init_mode == 'fixed'), gate_softmax=gate_softmax,
                     gate_source=gate_source, val_source=val_source,
                     use_ffn=use_ffn, use_bmm_gate=use_bmm_gate)
            for _ in range(n_layers)
        ])

        # 输出层
        self.ln_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 顶向下投影初始化为小值（逐渐引入反馈）
        for layer in self.layers:
            nn.init.normal_(layer.pcn_layer.W_td.weight, mean=0.0, std=0.01)
        if self.init_mode == 'fixed':
            # 诊断 step1 的两个根因修复：
            # S1/S4: W_res std=0.02 → 谱范数 ~0.6 → 12 层梯度衰减 316×。
            #        恒等初始化恢复信号/梯度直通（等价 Transformer 恒等残差）。
            for layer in self.layers:
                nn.init.eye_(layer.pcn_layer.W_res.weight)
                # S2: 门控死锁 —— w_g 输出尺度太小，g 恒 0.5。
                #     Xavier 匹配 [e⊙e, e+e] 输入维度，配合 gate_ln 使 g 初始有选择性
                nn.init.xavier_uniform_(layer.pcn_layer.w_g.weight)

    def forward(self, input_ids, topk=64, no_feedback=False, no_gating=False,
                return_stats=False, feedback_mode='prev_layer', n_pass=1):
        """
        input_ids: [B, N] token ids
        feedback_mode:
          'prev_layer' — 首轮实现：层 i 的 top-down 来自第 i-1 层（更浅层）。
              注：这不是预测编码意义上的 top-down（应为深层预测浅层），
              保留作消融对照臂。
          'two_pass'  — 真 top-down：pass1 无反馈前向存各层输出，
              pass2 中层 i 的预测来源是更深的第 i+1 层（顶层除外）。
              计算量 ×2，pass1 的状态 detach（推断近似，不反传）。
        n_pass: K-pass 迭代推断（V6 级别 2b）。n_pass=K>1 时执行 K 遍前向：
              pass1 无反馈；pass k (k≥2) 的层 i 用 pass k-1 的层 i+1 状态做
              top-down（误差松弛近似，W&B 等价性的推断条件）。
              末 pass 的误差 e（记录在 last_e）用于局部学习规则。
              优先于 feedback_mode 生效（n_pass=2 等价 two_pass）。
        """
        B, N = input_ids.shape

        # Embedding
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(N, device=input_ids.device))
        x = self.emb_dropout(x)

        all_stats = [] if return_stats else None

        if n_pass > 1 and not no_feedback:
            # K-pass 迭代推断（no_gating 只是聚合器消融，反馈/K-pass 照常适用）
            prev_states = None
            h_final = x
            for k in range(n_pass):
                states = []
                hk = x
                for i, layer in enumerate(self.layers):
                    if k == 0 or i + 1 >= len(self.layers):
                        h_above = None
                    else:
                        h_above = prev_states[i + 1].detach()
                    out = layer(hk, h_above=h_above, topk=topk,
                                no_feedback=no_feedback, no_gating=no_gating,
                                return_stats=return_stats)
                    if isinstance(out, tuple):
                        hk, stats = out
                        if return_stats:
                            all_stats.append(stats)
                    else:
                        hk = out
                    states.append(hk)
                prev_states = states
                h_final = hk
            x = h_final
        elif feedback_mode == 'two_pass' and not no_feedback and not no_gating:
            # pass 1：无反馈，收集各层输出
            hs = []
            h1 = x
            for layer in self.layers:
                out = layer(h1, h_above=None, topk=topk, return_stats=False)
                hs.append(out)
                h1 = out
            # pass 2：层 i 的 top-down 来自更深层 hs[i+1]
            h2 = x
            for i, layer in enumerate(self.layers):
                h_above = hs[i + 1].detach() if i + 1 < len(self.layers) else None
                out = layer(h2, h_above=h_above, topk=topk, return_stats=return_stats)
                if isinstance(out, tuple):
                    h2, stats = out
                    if return_stats:
                        all_stats.append(stats)
                else:
                    h2 = out
            x = h2
        else:
            # 逐层前向，传递 h_above
            prev_h = None
            for i, layer in enumerate(self.layers):
                # 反馈通路只在显式 no_feedback 时关闭；no_gating 是「聚合器消融」，保留反馈
                h_above = prev_h if (i > 0 and not no_feedback) else None
                out = layer(x, h_above=h_above, topk=topk,
                            no_feedback=no_feedback, no_gating=no_gating,
                            return_stats=return_stats)
                if isinstance(out, tuple):
                    x, stats = out
                    if return_stats:
                        all_stats.append(stats)
                else:
                    x = out
                prev_h = x  # 保存当前层输出作为下一层的 h_above

        # LM head
        x = self.ln_out(x)
        logits = self.lm_head(x)

        if return_stats and all_stats:
            return logits, all_stats
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
