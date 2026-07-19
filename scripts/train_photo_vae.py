import sys

sys.path.append("../src")
import torch
from datasets.bird_photos import BirdPhotos
from models.vae import VAEParameters
from train.train_photo_vae import train, PhotoVAETrainParameters
import time

DATASET_PATH = "../data/iNaturalistPrunedResizedPhoto"
vae_parameters = VAEParameters(
    input_channels=3,
    latent_dim=4,
    reconstruction_weight=1.0,
    kl_weight=0.001,
)
train_parameters = PhotoVAETrainParameters(
    lr=0.001,
    batch_size=128,
    epochs=30,
    num_workers=16,
    model_save_path="../models/photo_vae/" + str(time.time_ns()),
    model_save_name="photo_vae",
    model_save_frequency=1,
    device="cuda:0",
    seed=42,
    compile=True,
)


normalize_transform = lambda x: x.to(torch.float32) / 255
train_dataset = BirdPhotos(
    DATASET_PATH, split="train", photo_transform=normalize_transform
)
val_dataset = BirdPhotos(DATASET_PATH, split="val", photo_transform=normalize_transform)


train_outputs = train(
    train_dataset, val_dataset, None, vae_parameters, train_parameters
)
