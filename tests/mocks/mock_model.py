"""Reusable lightweight model mocks for EGX test suite."""

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """Minimal trainable model for kernel and pipeline tests."""

    def __init__(self, in_features: int = 16, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_ids=None, **kwargs):
        x = (
            input_ids.float()
            if input_ids is not None
            else kwargs.get("input", torch.zeros(1, 16))
        )
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[-1] != self.linear.in_features:
            x = torch.zeros(x.shape[0], self.linear.in_features)
        return self.linear(x)


class TinyModelWithLoss(nn.Module):
    """Model that returns an object with a .loss attribute."""

    def __init__(self, in_features: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, **kwargs):
        x = kwargs.get("input_ids", kwargs.get("input", torch.zeros(1, 16))).float()
        if x.shape[-1] != self.linear.in_features:
            x = torch.zeros(x.shape[0], self.linear.in_features)
        out = self.linear(x)

        class Output:
            def __init__(self, loss_val):
                self.loss = loss_val

        return Output(out.sum())


class ModelWithQProj(nn.Module):
    """Model with a q_proj layer for LoRA injection tests."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.out_proj = nn.Linear(64, 64)

    def forward(self, x):
        return self.out_proj(self.v_proj(self.q_proj(x)))
