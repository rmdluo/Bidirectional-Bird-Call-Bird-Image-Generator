import sys

sys.path.append("../src")
import torch
from datasets.bird_call_photo_latent_mix import BirdCallPhotoLatentMix
from models.diffusion import Diffusion
from models.vae import VAE
from metrics.photo_metrics import calculate_fid, calculate_accuracy
from processing.audio import log_mel_to_waveform
from IPython.display import Audio, display
import torchvision
from pathlib import Path
from tqdm import tqdm

AUDIO_DATASET_PATH = "../data/iNaturalistPrunedPreprocessedAudio"
PHOTO_DATASET_PATH = "../data/iNaturalistPrunedCropped"
AUDIO_LATENT_DATASET_PATH = "../data/iNaturalistLatentsAudio"
PHOTO_LATENT_DATASET_PATH = "../data/iNaturalistLatentsPhoto"

AUDIO_VAE_CHECKPOINT_PATH = (
    "../models/audio_vae/1784428657621265322/photo_vae_final.tar"
)
PHOTO_VAE_CHECKPOINT_PATH = (
    "../models/photo_vae/1784935571103765988/photo_vae_final.tar"
)
PHOTO_DIFFUSION_CHECKPOINT_PATH = "../models/conditional_photo_diffusion/1785017154428049849/conditional_photo_diffusion_final.tar"
AUDIO_DIFFUSION_CHECKPOINT_PATH = "../models/conditional_audio_diffusion/1785032242114494706/conditional_audio_diffusion_final.tar"

CONTRASTIVE_LATENT_DATASET_PATH = "../data/iNaturalistLatentsContrastiveAudio"
CONTRASTIVE_AUDIO_VAE_CHECKPOINT_PATH = "../models/contrastive_audio_vae/1785588335498284773/contrastive_audio_vae_final.tar"
CONTRASTIVE_PHOTO_DIFFUSION_CHECKPOINT_PATH = "../models/contrastive_conditional_photo_diffusion/1785617075648407339/contrastive_conditional_photo_diffusion_final.tar"
CONTRASTIVE_AUDIO_DIFFUSION_CHECKPOINT_PATH = "../models/contrastive_conditional_audio_diffusion/1785615911625265945/contrastive_conditional_audio_diffusion_final.tar"

PHOTO_CLASSIFIER_PATH = "../models/yolo26_bird_classifier/weights/best.pt"

GENERATED_PHOTO_PATH = "../data/GeneratedPhoto"
GENERATED_AUDIO_PATH = "../data/GeneratedAudio"
DEVICE = "cuda:0"
BATCH_SIZE = 32


def evaluate_audio_metrics():
    pass


def evaluate_photo_metrics():
    results = {}

    for contrastive in [False, True]:
        diffusion_checkpoint = torch.load(
            PHOTO_DIFFUSION_CHECKPOINT_PATH, weights_only=False, map_location="cpu"
        )
        diffusion_parameters = diffusion_checkpoint["model_parameters"]
        train_parameters = diffusion_checkpoint["train_parameters"]
        train_parameters.device = DEVICE

        diffusion = Diffusion(diffusion_parameters)
        diffusion.load_state_dict(diffusion_checkpoint["model_state_dict"])
        diffusion = diffusion.to(train_parameters.device)
        diffusion.eval()

        vae_checkpoint = torch.load(
            PHOTO_VAE_CHECKPOINT_PATH, weights_only=False, map_location="cpu"
        )
        vae_parameters = vae_checkpoint["model_parameters"]
        vae = VAE(vae_parameters)
        vae.load_state_dict(vae_checkpoint["model_state_dict"])
        vae = vae.to(train_parameters.device)
        vae.eval()

        train_dataset = BirdCallPhotoLatentMix(
            (
                AUDIO_LATENT_DATASET_PATH
                if not contrastive
                else CONTRASTIVE_LATENT_DATASET_PATH
            ),
            PHOTO_LATENT_DATASET_PATH,
            (
                AUDIO_VAE_CHECKPOINT_PATH
                if not contrastive
                else CONTRASTIVE_AUDIO_VAE_CHECKPOINT_PATH
            ),
            PHOTO_VAE_CHECKPOINT_PATH,
            split="train",
            audio_first=False,
            num_pairs_per_species=50,
        )

        val_dataset = BirdCallPhotoLatentMix(
            (
                AUDIO_LATENT_DATASET_PATH
                if not contrastive
                else CONTRASTIVE_LATENT_DATASET_PATH
            ),
            PHOTO_LATENT_DATASET_PATH,
            (
                AUDIO_VAE_CHECKPOINT_PATH
                if not contrastive
                else CONTRASTIVE_AUDIO_VAE_CHECKPOINT_PATH
            ),
            PHOTO_VAE_CHECKPOINT_PATH,
            split="val",
            audio_first=False,
            num_pairs_per_species=50,
        )

        save_path = Path(GENERATED_PHOTO_PATH) / (
            "Contrastive" if contrastive else "Noncontrastive"
        )

        for split, dataset in zip(["train", "val"], [train_dataset, val_dataset]):
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
            )

            for i, (latent_1, latent_2, species) in enumerate(dataloader):
                if (
                    i % 100 != 0
                    or (save_path / split / species[0] / f"generated_{i}.jpg").exists()
                ):
                    continue
                print(f"{i}/{len(dataset)}", end="\r", flush=True)
                latent_1 = latent_1.to(DEVICE)
                latent_2 = latent_2.to(DEVICE)

                samples = diffusion.sample(latent_2, dif_steps=50, device=DEVICE)

                with torch.no_grad():
                    photo_generated = vae.decode(samples[-1].to(DEVICE))
                    (save_path / split / species[0]).mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(
                        photo_generated,
                        save_path / split / species[0] / f"generated_{i}.jpg",
                    )
            print()

            fid = calculate_fid(str(save_path / split), PHOTO_DATASET_PATH)
            accuracy = calculate_accuracy(
                save_path,
                split=split,
                yolo_path=PHOTO_CLASSIFIER_PATH,
                batch_size=BATCH_SIZE * 32,
                device=DEVICE,
            )

            results[contrastive] = {
                split: {
                    "fid": fid,
                    "accuracy": accuracy,
                }
            }

    return results


print(evaluate_photo_metrics())
