import warnings
from typing import Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_mean   = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder reconstructs the input from a latent sample
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.ReLU()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.activation(self.encoder_hidden(x))
        return self.encoder_mean(h), self.encoder_logvar(h)

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
     
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.decoder_hidden(z))
        return self.decoder_output(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterise(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar


def _vae_loss(reconstruction: torch.Tensor,
              x: torch.Tensor,
              mean: torch.Tensor,
              logvar: torch.Tensor,
              beta: float = 1.0) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def augment_with_vae(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_ratio: float = 2.0,
    epochs: int = 200,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_original = len(X_train)
    n_synthetic = int(n_original * (target_ratio - 1))
    if n_synthetic <= 0:
        return X_train, y_train

    try:
        # Build joint training matrix: features + target as last column
        X_np = X_train.values.astype(np.float32)
        y_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        joint = np.concatenate([X_np, y_np], axis=1)

        # Standardise (VAE trains far more stably on scaled data)
        scaler = StandardScaler()
        joint_scaled = scaler.fit_transform(joint)

        # Train the VAE
        input_dim = joint_scaled.shape[1]
        vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        data_tensor = torch.tensor(joint_scaled, dtype=torch.float32)

        vae.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            reconstruction, mean, logvar = vae(data_tensor)
            loss = _vae_loss(reconstruction, data_tensor, mean, logvar)
            loss.backward()
            optimizer.step()

        vae.eval()
        with torch.no_grad():
            z_samples = torch.randn(n_synthetic, latent_dim)
            synth_scaled = vae.decode(z_samples).numpy()

        # De-standardise back to original scale
        synth = scaler.inverse_transform(synth_scaled)
        synth_X = synth[:, :-1]
        synth_y = synth[:, -1]

    except Exception as e:
        warnings.warn(
            f"VAE augmentation failed: {type(e).__name__}: {e}. "
            f"Returning original training data unchanged."
        )
        return X_train, y_train

    # Combine original + synthetic
    X_synth_df = pd.DataFrame(synth_X, columns=X_train.columns)
    X_augmented = pd.concat([X_train.reset_index(drop=True), X_synth_df], ignore_index=True)
    y_augmented = np.concatenate([y_train, synth_y])

    return X_augmented, y_augmented