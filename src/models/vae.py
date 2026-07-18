import torch
from dataclasses import dataclass
from models.common import ResidualBlock


@dataclass
class VAEParameters:
    latent_dim: int = 8
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.0001


@dataclass
class VAEOutput:
    original_input: torch.Tensor
    reconstructed: torch.Tensor
    sampled_latent: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


@dataclass
class VAELoss:
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor


class VAE(torch.nn.Module):
    def __init__(self, params: VAEParameters):
        super().__init__()

        latent_dim = params.latent_dim
        reconstruction_weight = params.reconstruction_weight
        kl_weight = params.kl_weight

        # Assume input size of 3 x 128 x 128
        # Get output size of latent_dim x 16 x 16
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, padding=3),
            ResidualBlock(64, 64, kernel_size=3),  # 64 x 128 x 128
            ResidualBlock(64, 64, kernel_size=3),  # 64 x 128 x 128
            torch.nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # 128 x 64 x 64
            ResidualBlock(128, 128, kernel_size=3),  # 128 x 64 x 64
            ResidualBlock(128, 128, kernel_size=3),  # 128 x 64 x 64
            torch.nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1
            ),  # 256 x 32 x 32
            ResidualBlock(256, 256, kernel_size=3),  # 256 x 32 x 32
            ResidualBlock(256, 256, kernel_size=3),  # 256 x 32 x 32
            torch.nn.Conv2d(
                256, 256, kernel_size=3, stride=2, padding=1
            ),  # 256 x 16 x 16
            torch.nn.GroupNorm(8, 256),  # 256 x 16 x 16
            torch.nn.SiLU(),  # 256 x 16 x 16
        )
        self.encoder_mu = torch.nn.Conv2d(256, latent_dim, 3, padding="same")
        self.encoder_logvar = torch.nn.Conv2d(256, latent_dim, 3, padding="same")

        # Assume input size of latent_dim x 16 x 16
        # Get output size of 3 x 128 x 128
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, 256, 3, padding="same"),  # 256 x 16 x 16
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),  # 256 x 32 x 32
            ResidualBlock(256, 256),  # 256 x 32 x 32
            ResidualBlock(256, 256),  # 256 x 32 x 32
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),  # 256 x 64 x 64
            torch.nn.Conv2d(256, 128, 3, padding="same"),  # 128 x 64 x 64
            ResidualBlock(128, 128),  # 128 x 64 x 64
            ResidualBlock(128, 128),  # 128 x 64 x 64
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),  # 128 x 128 x 128
            torch.nn.Conv2d(128, 64, 3, padding="same"),  # 64 x 128 x 128
            ResidualBlock(64, 64),  # 64 x 128 x 128
            ResidualBlock(64, 64),  # 64 x 128 x 128
        )
        self.decoder_final = torch.nn.Sequential(
            torch.nn.GroupNorm(8, 64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 3, 7, padding="same"),
        )

        # Param for loss
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

    def encode(self, x):
        z = self.encoder(x)
        return self.encoder_mu(z), self.encoder_logvar(z)

    def decode(self, z):
        x = self.decoder(z)
        return self.decoder_final(x)

    @staticmethod
    def sample(mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        return mu + std_dev * torch.randn_like(std_dev)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        reconstructed = self.decode(z)
        return VAEOutput(x, reconstructed, z, mu, logvar)

    def loss(self, vae_output: VAEOutput):
        # See https://docs.pytorch.org/docs/2.13/generated/torch.nn.L1Loss.html
        reconstruction_loss = torch.nn.functional.l1_loss(
            vae_output.reconstructed, vae_output.original_input
        )
        kl_loss = -0.5 * torch.mean(
            1
            + vae_output.logvar
            - torch.pow(vae_output.mu, 2)
            - torch.exp(vae_output.logvar)
        )

        return VAELoss(
            self.reconstruction_weight * reconstruction_loss + self.kl_weight * kl_loss,
            reconstruction_loss,
            kl_loss,
        )
