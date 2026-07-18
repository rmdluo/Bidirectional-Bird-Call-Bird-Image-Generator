import torch
from dataclasses import dataclass
from models.vae import VAE, VAEParameters, VAEOutput, VAELoss


@dataclass
class PhotoVAETrainParameters:
    lr: float = 0.001
    batch_size: int = 32


def train(
    photo_dataset,
    vae_parameters: VAEParameters,
    train_parameters: PhotoVAETrainParameters,
):
    pass
