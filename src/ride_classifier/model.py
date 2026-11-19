"""1D convolutional network that classifies a window of sensor measurements."""

from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Convolution -> normalisation -> activation -> downsampling."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class RideCNN(nn.Module):
    """Input: (batch, time steps, sensors). Output: one score per class."""

    def __init__(self, n_sensors: int = 6, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(n_sensors, 32, kernel_size=7),
            conv_block(32, 64, kernel_size=5),
            conv_block(64, 64, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),  # average over time: works for any window length
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # Conv1d expects (batch, channels, time)
        return self.classifier(self.features(x))
