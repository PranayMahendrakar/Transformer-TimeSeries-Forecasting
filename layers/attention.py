"""
Attention mechanisms: Full Self-Attention, ProbSparse Attention (Informer),
and Auto-Correlation (Autoformer).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FullAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, mask_flag: bool = True, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A if self.output_attention else None


class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention from Informer (AAAI 2021).
    Reduces complexity from O(L^2) to O(L log L) by selecting
    the top-u dominant queries via a sparsity measurement.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 5,
                 scale: Optional[float] = None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor,
                 sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample K indices
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Sparsity measurement: max - mean
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use reduced Q for full attention
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in: torch.Tensor, V: torch.Tensor,
                        scores: torch.Tensor, index: torch.Tensor,
                        L_Q: int, attn_mask: Optional[torch.Tensor]
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, L_V, D = V.shape
        if self.mask_flag and attn_mask is not None:
            attn_mask = torch.tril(torch.ones(index.shape[-1], L_V,
                                              device=V.device), diagonal=0).bool()
            scores.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in, (attn if self.output_attention else None)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * math.ceil(math.log(L_K))
        u = self.factor * math.ceil(math.log(L_Q))
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / math.sqrt(D)
        scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.contiguous().transpose(2, 1), attn


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism from Autoformer (NeurIPS 2021).
    Discovers period-based dependencies and aggregates top-k similar sub-series.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 1,
                 scale: Optional[float] = None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values: torch.Tensor,
                                corr: torch.Tensor) -> torch.Tensor:
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Find top-k periods
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)

        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1)
                                                  .unsqueeze(1).unsqueeze(1).repeat(
                1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor,
                                 corr: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)            .repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)                .repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1)
                                                  .unsqueeze(1).unsqueeze(1).repeat(
                1, head, channel, length))
        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # FFT-based auto-correlation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        return V.contiguous(), corr.permute(0, 3, 1, 2) if self.output_attention else None


class AttentionLayer(nn.Module):
    """
    Wrapper that projects Q, K, V and applies any attention mechanism.
    """

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys: Optional[int] = None, d_values: Optional[int] = None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
