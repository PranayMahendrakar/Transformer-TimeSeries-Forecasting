"""
Series Decomposition blocks used by Autoformer.
Separates a time series into trend and seasonal components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """
    Moving average block to extract the trend component.
    Uses a 1D average pooling with padding to preserve sequence length.
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad both ends to preserve input length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block.
    Returns (seasonal, trend) components.
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean      # seasonal component
        return residual, moving_mean    # (seasonal, trend)


class MultiScaleDecomposition(nn.Module):
    """
    Multi-scale series decomposition using multiple kernel sizes.
    Averages trends across all scales.
    """

    def __init__(self, kernel_sizes=(13, 25)):
        super(MultiScaleDecomposition, self).__init__()
        self.decomps = nn.ModuleList([SeriesDecomposition(k) for k in kernel_sizes])

    def forward(self, x: torch.Tensor):
        seasonals, trends = zip(*[d(x) for d in self.decomps])
        trend = torch.stack(trends, dim=0).mean(dim=0)
        seasonal = x - trend
        return seasonal, trend
