"""Opacity prediction head leveraging UniCeption's DPT regression decoder."""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from uniception.models.prediction_heads.base import PredictionHeadInput
from uniception.models.prediction_heads.dpt import DPTRegressionProcessor


@dataclass
class OpacityHeadOutput:
    """Container for opacity logits and probabilities."""

    probability: torch.Tensor
    logits: torch.Tensor


class OpacityHead(nn.Module):
    """Predict per-pixel opacity volumes using UniCeption's DPT regression blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        checkpoint_gradient: bool = False,
    ):
        """Initialise the opacity decoder backed by UniCeption's DPT processor.

        Args:
            input_dim: Channel dimension of the incoming DPT decoded features.
            hidden_dim: Unused compatibility parameter retained for config stability.
            output_dim: Number of temporal opacity bins ``T`` to regress per pixel.
            checkpoint_gradient: Whether to enable gradient checkpointing inside the
                UniCeption regression processor.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder = DPTRegressionProcessor(
            input_feature_dim=input_dim,
            output_dim=output_dim,
            checkpoint_gradient=checkpoint_gradient,
        )

    def forward(
        self, features: Union[torch.Tensor, PredictionHeadInput]
    ) -> OpacityHeadOutput:
        """Decode temporally-indexed opacity logits and probabilities.

        Args:
            features: Either a dense feature tensor of shape ``(B, C, H, W)`` or a
                ``PredictionHeadInput`` whose ``last_feature`` field contains such a
                tensor.

        Returns:
            ``OpacityHeadOutput`` whose tensors have shape ``(B, T, H, W)``.
        """

        if isinstance(features, PredictionHeadInput):
            head_input = features
        else:
            head_input = PredictionHeadInput(last_feature=features)

        decoder_output = self.decoder(head_input)
        logits = decoder_output.decoded_channels
        probability = torch.sigmoid(logits)
        return OpacityHeadOutput(probability=probability, logits=logits)


__all__ = ["OpacityHead", "OpacityHeadOutput"]
