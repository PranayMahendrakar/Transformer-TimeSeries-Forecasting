"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
NeurIPS 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from layers.attention import AutoCorrelation, AttentionLayer
from layers.decomposition import SeriesDecomposition
from layers.embedding import DataEmbedding_wo_pos


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attn
        self.cross_attention = cross_attn
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(d_model, c_out, 3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        trend = trend1 + trend2 + trend3
        return x, self.projection(trend.permute(0, 2, 1)).transpose(1, 2)


class AutoformerEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm:
            x = self.norm(x)
        return x, attns


class AutoformerDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, trend=None, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, residual = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual
        if self.norm:
            x = self.norm(x)
        if self.projection:
            x = self.projection(x)
        return x, trend


class Autoformer(nn.Module):
    """
    Autoformer for long-term forecasting.

    Args:
        enc_in, dec_in, c_out: Feature dimensions
        seq_len:    Input sequence length
        label_len:  Decoder overlap token length
        pred_len:   Prediction horizon
        d_model:    Model dimension (default 512)
        n_heads:    Attention heads (default 8)
        e_layers:   Encoder layers (default 2)
        d_layers:   Decoder layers (default 1)
        d_ff:       FFN dimension (default 2048)
        moving_avg: Trend extraction kernel size (default 25)
        factor:     Auto-Correlation factor (default 1)
        dropout:    Dropout rate (default 0.05)
        embed:      Embedding type (default 'timeF')
        freq:       Time frequency (default 'h')
        activation: 'relu' or 'gelu' (default 'gelu')
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 moving_avg=25, factor=1, dropout=0.05, embed='timeF', freq='h',
                 activation='gelu', output_attention=False):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.decomp = SeriesDecomposition(moving_avg)
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        self.encoder = AutoformerEncoder(
            [AutoformerEncoderLayer(
                AttentionLayer(AutoCorrelation(False, factor, attention_dropout=dropout,
                                               output_attention=output_attention), d_model, n_heads),
                d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.decoder = AutoformerDecoder(
            [AutoformerDecoderLayer(
                AttentionLayer(AutoCorrelation(True, factor, attention_dropout=dropout,
                                               output_attention=False), d_model, n_heads),
                AttentionLayer(AutoCorrelation(False, factor, attention_dropout=dropout,
                                               output_attention=False), d_model, n_heads),
                d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
            ) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(x_dec.shape[0], self.pred_len, x_dec.shape[2], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal, trend = self.decoder(dec_out, enc_out, trend=trend_init,
                                       x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        out = trend + seasonal
        if self.output_attention:
            return out[:, -self.pred_len:, :], attns
        return out[:, -self.pred_len:, :]

