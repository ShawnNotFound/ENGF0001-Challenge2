#!/usr/bin/env python3
"""
Reusable sequence model components shared between training and live inference.
"""

import torch
from torch import nn


class LSTMAnomalyDetector(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        seq_out, _ = self.lstm(x)
        last_hidden = seq_out[:, -1, :]
        return self.head(last_hidden)


def build_lstm_model(config: dict):
    return LSTMAnomalyDetector(
        input_size=config["input_size"],
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        output_dim=config["output_dim"],
    )


def load_lstm_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model = build_lstm_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
