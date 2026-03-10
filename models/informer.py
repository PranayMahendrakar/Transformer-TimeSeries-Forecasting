"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
AAAI 2021 Best Paper Award

Key innovations:
  - ProbSparse self-attention: O(L log L) complexity
  - Self-attention distilling: halves sequence length each layer
  - Generative decoder for direct long-sequence prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from layers.attention import ProbSparseAttention, FullAttention, AttentionLayer
from layers.embedding import DataEmbedding


class ConvLayer(nn.Module):
    """Distilling convolution: MaxPool halves sequence length."""

    def __init__(self, c_in: int):
        super().__init__()
        self.downConv = nn.Conv1d(c_in, c_in, kernel_size=3,
                                  padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn, conv in zip(self.attn_layers, self.conv_layers):
                x, a = attn(x, attn_mask=attn_mask)
                x = conv(x)
                attns.append(a)
            x, a = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(a)
        else:
            for attn in self.attn_layers:
                x, a = attn(x, attn_mask=attn_mask)
                attns.append(a)
        if self.norm:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff=None,
                 dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attn
        self.cross_attention = cross_attn
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm:
            x = self.norm(x)
        if self.projection:
            x = self.projection(x)
        return x


class Informer(nn.Module):
    """
    Informer model for long-sequence time series forecasting.

    Args:
        enc_in:    Encoder input feature count
        dec_in:    Decoder input feature count
        c_out:     Output feature count
        seq_len:   Encoder sequence length
        label_len: Decoder start-token length
        out_len:   Prediction horizon
        d_model:   Model dimension (default 512)
        n_heads:   Attention heads (default 8)
        e_layers:  Encoder layers (default 3)
        d_layers:  Decoder layers (default 2)
        d_ff:      Feed-forward dim (default 512)
        factor:    ProbSparse factor (default 5)
        attn:      'prob' or 'full' (default 'prob')
        embed:     Embedding type: 'fixed'|'learned'|'timeF'
        freq:      Time frequency: 'h'|'t'|'s'|'m'|'d'
        dropout:   Dropout rate
        distil:    Use distilling encoder (default True)
        activation: 'relu' or 'gelu'
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 factor=5, attn='prob', embed='fixed', freq='h', dropout=0.05,
                 distil=True, activation='gelu', output_attention=False):
        super().__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        Attn = ProbSparseAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(Attn(False, factor, attention_dropout=dropout,
                                    output_attention=output_attention), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [DecoderLayer(
                AttentionLayer(FullAttention(True, factor, attention_dropout=dropout,
                                             output_attention=False), d_model, n_heads),
                AttentionLayer(FullAttention(False, factor, attention_dropout=dropout,
                                             output_attention=False), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]

