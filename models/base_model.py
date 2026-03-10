"""
Base model class for all time series forecasting models.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseForecaster(nn.Module, ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, c_out: int):
        super(BaseForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out

    @abstractmethod
    def forward(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None,
                x_dec: Optional[torch.Tensor] = None, x_mark_dec: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_enc: Encoder input  [Batch, SeqLen, Features]
            x_mark_enc: Encoder time features [Batch, SeqLen, TimeFeatures]
            x_dec: Decoder input  [Batch, LabelLen+PredLen, Features]
            x_mark_dec: Decoder time features
        Returns:
            Predictions [Batch, PredLen, c_out]
        """
        pass

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"seq_len={self.seq_len}, pred_len={self.pred_len}, "
                f"enc_in={self.enc_in}, c_out={self.c_out}, "
                f"params={self.count_parameters():,})")
