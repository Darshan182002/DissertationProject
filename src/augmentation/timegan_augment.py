import warnings
from typing import Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 32, output_dim: int = 72):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 8, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def _train_timegan(
    data_tensor: torch.Tensor,
    input_dim: int,
    hidden_dim: int = 32,
    latent_dim: int = 8,
    noise_dim: int = 8,
    ae_epochs: int = 100,
    gen_epochs: int = 100,
    joint_epochs: int = 100,
    lr: float = 1e-3,
) -> Tuple[Encoder, Decoder, Generator]:
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    generator = Generator(noise_dim, hidden_dim, latent_dim)
    discriminator = Discriminator(latent_dim, hidden_dim)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # Phase 1: autoencoder pretraining
    ae_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )
    for _ in range(ae_epochs):
        ae_optimizer.zero_grad()
        z = encoder(data_tensor)
        x_recon = decoder(z)
        loss = mse_loss(x_recon, data_tensor)
        loss.backward()
        ae_optimizer.step()

    # Phase 2: generator adversarial training
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    real_labels = torch.ones(data_tensor.shape[0], 1)
    fake_labels = torch.zeros(data_tensor.shape[0], 1)

    for _ in range(gen_epochs):
        with torch.no_grad():
            real_latents = encoder(data_tensor)
        noise = torch.randn(data_tensor.shape[0], noise_dim)
        fake_latents = generator(noise)

        disc_optimizer.zero_grad()
        d_real = discriminator(real_latents)
        d_fake = discriminator(fake_latents.detach())
        d_loss = bce_loss(d_real, real_labels) + bce_loss(d_fake, fake_labels)
        d_loss.backward()
        disc_optimizer.step()

        # Train generator: fool discriminator into thinking fakes are real
        gen_optimizer.zero_grad()
        d_fake_for_g = discriminator(fake_latents)
        g_loss = bce_loss(d_fake_for_g, real_labels)
        g_loss.backward()
        gen_optimizer.step()

    joint_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters())
        + list(generator.parameters()),
        lr=lr * 0.5,
    )
    for _ in range(joint_epochs):
        joint_optimizer.zero_grad()
        z = encoder(data_tensor)
        x_recon = decoder(z)
        recon_loss = mse_loss(x_recon, data_tensor)

        noise = torch.randn(data_tensor.shape[0], noise_dim)
        fake_latents = generator(noise)
        fake_recon = decoder(fake_latents)
        adv_loss = mse_loss(fake_recon.mean(dim=0), data_tensor.mean(dim=0))

        total_loss = recon_loss + 0.1 * adv_loss
        total_loss.backward()
        joint_optimizer.step()

    return encoder, decoder, generator


def augment_with_timegan(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_ratio: float = 2.0,
    ae_epochs: int = 100,
    gen_epochs: int = 100,
    joint_epochs: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_original = len(X_train)
    n_synthetic = int(n_original * (target_ratio - 1))
    if n_synthetic <= 0:
        return X_train, y_train

    try:
        # Build joint training matrix and standardise
        X_np = X_train.values.astype(np.float32)
        y_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        joint = np.concatenate([X_np, y_np], axis=1)

        scaler = StandardScaler()
        joint_scaled = scaler.fit_transform(joint).astype(np.float32)

        data_tensor = torch.tensor(joint_scaled, dtype=torch.float32)

        # Train the TimeGAN
        encoder, decoder, generator = _train_timegan(
            data_tensor=data_tensor,
            input_dim=joint_scaled.shape[1],
            ae_epochs=ae_epochs,
            gen_epochs=gen_epochs,
            joint_epochs=joint_epochs,
        )

        encoder.eval()
        decoder.eval()
        generator.eval()

        with torch.no_grad():
            noise = torch.randn(n_synthetic, 8)  # noise_dim = 8
            fake_latents = generator(noise)
            synth_scaled = decoder(fake_latents).numpy()

        synth = scaler.inverse_transform(synth_scaled)
        synth_X = synth[:, :-1]
        synth_y = synth[:, -1]

    except Exception as e:
        warnings.warn(
            f"TimeGAN augmentation failed: {type(e).__name__}: {e}. "
            f"Returning original training data unchanged."
        )
        return X_train, y_train

    # Combine original + synthetic
    X_synth_df = pd.DataFrame(synth_X, columns=X_train.columns)
    X_augmented = pd.concat([X_train.reset_index(drop=True), X_synth_df], ignore_index=True)
    y_augmented = np.concatenate([y_train, synth_y])

    return X_augmented, y_augmented