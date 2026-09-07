import warnings
from typing import Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleLSTM(nn.Module):

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take the last time step's output
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


def _make_windows(series: np.ndarray, lookback: int) -> tuple:

    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32)
    return torch.tensor(X), torch.tensor(y)


def fit_predict_lstm(X_train: pd.DataFrame,
                     y_train: np.ndarray,
                     X_test: pd.DataFrame,
                     lookback: int = 8,
                     hidden_dim: int = 32,
                     epochs: int = 200,
                     lr: float = 1e-3,
                     seed: int = 42,
                     ) -> np.ndarray:

    torch.manual_seed(seed)
    np.random.seed(seed)

    horizon = len(X_test)

    if len(y_train) < lookback + 2:
        # Fold has too little data — fall back to mean prediction
        warnings.warn(
            f"LSTM training set too short ({len(y_train)} < {lookback + 2}). "
            f"Returning mean prediction."
        )
        return np.full(horizon, float(np.mean(y_train)))

    # Standardise the series (LSTMs train far more stably on scaled data)
    train_mean = y_train.mean()
    train_std = y_train.std()
    if train_std < 1e-8:
        # Degenerate constant series
        return np.full(horizon, float(train_mean))
    y_train_scaled = (y_train - train_mean) / train_std

    # Build sliding windows from training data
    X_windows, y_windows = _make_windows(y_train_scaled, lookback)

    # Train LSTM
    model = SimpleLSTM(input_dim=1, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_windows)
        loss = loss_fn(y_pred, y_windows)
        loss.backward()
        optimizer.step()

    # Iterative forecasting: start with the last `lookback` training values
    model.eval()
    context = torch.tensor(y_train_scaled[-lookback:].astype(np.float32)).reshape(1, lookback, 1)

    predictions_scaled = []
    with torch.no_grad():
        for _ in range(horizon):
            next_pred = model(context).item()
            predictions_scaled.append(next_pred)
            # Slide the window: drop first, append predicted value
            new_context = torch.cat([context[:, 1:, :],
                                      torch.tensor([[[next_pred]]], dtype=torch.float32)],
                                     dim=1)
            context = new_context

    # De-standardise back to original scale
    predictions = np.array(predictions_scaled) * train_std + train_mean
    return predictions