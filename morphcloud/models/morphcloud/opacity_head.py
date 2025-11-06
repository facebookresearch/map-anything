"""Lightweight decoder head for predicting per-pixel opacity volumes."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class OpacityHeadOutput:
    """Container for opacity logits and probabilities."""

    probability: torch.Tensor
    logits: torch.Tensor


class OpacityHead(nn.Module):
    """Predicts per-pixel opacity probabilities from dense DPT features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> OpacityHeadOutput:
        logits = self.network(features)
        probability = torch.sigmoid(logits)
        return OpacityHeadOutput(probability=probability, logits=logits)


__all__ = ["OpacityHead", "OpacityHeadOutput"]
