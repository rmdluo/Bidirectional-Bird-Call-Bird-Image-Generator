import sys

sys.path.append("../src")
from datasets.bird_call_photo_latent_mix import BirdCallPhotoLatentMix
from models.diffusion import DiffusionParameters
from train.train_diffusion import train, continue_training, DiffusionTrainParameters
import time

AUDIO_DATASET_PATH = "../data/iNaturalistLatentsAudio"
PHOTO_DATASET_PATH = "../data/iNaturalistLatentsPhoto"
AUDIO_VAE_CHECKPOINT_PATH = (
    "../models/audio_vae/1784428657621265322/photo_vae_final.tar"
)
PHOTO_VAE_CHECKPOINT_PATH = (
    "../models/photo_vae/1784935571103765988/photo_vae_final.tar"
)

diffusion_parameters = DiffusionParameters(
    input_shape=(4, 16, 16),
    conditioning_shape=(4, 16, 40),
    output_shape=(4, 16, 16),
    conditioning_dim=1024,
    unet_block_out_channels=(128, 256, 512),
    num_train_steps=1000,
    seed=42,
)
train_parameters = DiffusionTrainParameters(
    lr=0.0001,
    batch_size=128,
    epochs=200,
    num_workers=16,
    model_save_path="../models/conditional_photo_diffusion/" + str(time.time_ns()),
    model_save_name="conditional_photo_diffusion",
    model_save_frequency=10,
    device="cuda:0",
    seed=42,
    compile=True,
)


train_dataset = BirdCallPhotoLatentMix(
    AUDIO_DATASET_PATH,
    PHOTO_DATASET_PATH,
    AUDIO_VAE_CHECKPOINT_PATH,
    PHOTO_VAE_CHECKPOINT_PATH,
    split="train",
    audio_first=False,
    num_pairs_per_species=50,
)
val_dataset = BirdCallPhotoLatentMix(
    AUDIO_DATASET_PATH,
    PHOTO_DATASET_PATH,
    AUDIO_VAE_CHECKPOINT_PATH,
    PHOTO_VAE_CHECKPOINT_PATH,
    split="val",
    audio_first=False,
    num_pairs_per_species=50,
)


train_outputs = train(
    train_dataset, val_dataset, None, diffusion_parameters, train_parameters
)

### Uncomment to continue training
# train_outputs = continue_training(
#     "../models/conditional_photo_diffusion/1784978383724670836/conditional_photo_diffusion_epoch_9.tar",
#     train_dataset,
#     val_dataset,
#     None,
# )
