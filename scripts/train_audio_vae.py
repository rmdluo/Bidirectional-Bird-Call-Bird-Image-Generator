import sys

sys.path.append("../src")
from datasets.bird_audio import BirdAudio
from models.vae import VAEParameters
from train.train_photo_vae import train, PhotoVAETrainParameters
import time

DATASET_PATH = "../data/iNaturalistPrunedPreprocessedAudio"
vae_parameters = VAEParameters(
    input_channels=1,
    input_shape=(1, 128, 313),
    latent_dim=4,
    reconstruction_weight=1.0,
    kl_weight=0.001,
)
train_parameters = PhotoVAETrainParameters(
    lr=0.001,
    batch_size=32,
    epochs=30,
    num_workers=16,
    model_save_path="../models/audio_vae/" + str(time.time_ns()),
    model_save_name="audio_vae",
    model_save_frequency=1,
    device="cuda:1",
    seed=42,
    compile=True,
)


train_dataset = BirdAudio(DATASET_PATH, split="train", from_spectrogram=True)
val_dataset = BirdAudio(DATASET_PATH, split="val", from_spectrogram=True)


train_outputs = train(
    train_dataset, val_dataset, None, vae_parameters, train_parameters
)
