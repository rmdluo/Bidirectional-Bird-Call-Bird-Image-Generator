import sys

sys.path.append("../src")
from pathlib import Path
import torch
import torchvision
from tqdm import tqdm
from models.vae import VAE

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

DATASET_PATH = Path("../data/iNaturalistPrunedPreprocessedAudio")
CHECKPOINT_PATH = "../models/audio_vae/1784428657621265322/photo_vae_final.tar"
OUTPUT_PATH = Path("../data/iNaturalistLatentsAudio")

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

vae_parameters = checkpoint["model_parameters"]
train_parameters = checkpoint["train_parameters"]
train_parameters.device = device


normalize_transform = lambda x: torch.unsqueeze(x.to(torch.float32) / 255, 0).to(device)

vae = VAE(vae_parameters)
vae.load_state_dict(checkpoint["model_state_dict"])
vae = vae.to(train_parameters.device)
vae.eval()

files = list(DATASET_PATH.glob("**/*.pt"))

with torch.no_grad():
    for file in tqdm(files):
        log_mel = torch.load(file, weights_only=True).unsqueeze(0).to(device)

        mean, logvar = vae.encode(log_mel)

        output_path = (OUTPUT_PATH / file.relative_to(DATASET_PATH)).with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean": mean.cpu().squeeze(0),
                "logvar": logvar.cpu().squeeze(0),
            },
            output_path,
        )
