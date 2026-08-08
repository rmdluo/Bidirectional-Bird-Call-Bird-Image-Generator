import sys

sys.path.append("../src")
import torch
from datasets.bird_call_photo_mix import BirdCallPhotoMix
from models.vae import VAEParameters
from train.train_contrastive_audio_vae import train, ContrastiveVAETrainParameters
import time

AUDIO_DATASET_PATH = "../data/iNaturalistPrunedPreprocessedAudio"
PHOTO_DATASET_PATH = "../data/iNaturalistPrunedCropped"
PHOTO_VAE_CHECKPOINT_PATH = "../models/photo_vae/photo_only/photo_vae_final.tar"
vae_parameters = VAEParameters(
    input_channels=1,
    input_shape=(1, 128, 313),
    output_shape=(16, 16),
    latent_dim=4,
    reconstruction_weight=1.0,
    kl_weight=0.001,
    seed=42,
)
train_parameters = ContrastiveVAETrainParameters(
    lr=0.0001,
    batch_size=32,
    epochs=30,
    num_workers=16,
    infonce_logvar_weight=0.0,
    infonce_temperature=0.2,
    latent_mean_mae_loss_weight=1.0,
    latent_logvar_mae_loss_weight=1.0,
    contrastive_loss_weight=1.0,
    model_save_path="../models/contrastive_audio_vae/" + str(time.time_ns()),
    model_save_name="contrastive_audio_vae",
    model_save_frequency=20,
    device="cuda:0",
    seed=42,
    compile=False,
)


normalize_transform = lambda x: x.to(torch.float32) / 255
train_dataset = BirdCallPhotoMix(
    AUDIO_DATASET_PATH,
    PHOTO_DATASET_PATH,
    split="train",
    from_spectrogram=True,
    photo_transform=normalize_transform,
)
val_dataset = BirdCallPhotoMix(
    AUDIO_DATASET_PATH,
    PHOTO_DATASET_PATH,
    split="val",
    from_spectrogram=True,
    photo_transform=normalize_transform,
)


train_outputs = train(
    train_dataset,
    val_dataset,
    None,
    vae_parameters,
    PHOTO_VAE_CHECKPOINT_PATH,
    train_parameters,
)
