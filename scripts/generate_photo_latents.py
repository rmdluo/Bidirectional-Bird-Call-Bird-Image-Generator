import sys

sys.path.append("../src")
from pathlib import Path
import torch
import torchvision
from tqdm import tqdm
from models.vae import VAE

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

DATASET_PATH = Path("../data/iNaturalistPrunedResizedPhoto")
CHECKPOINT_PATH = "../models/photo_vae/1784428654983751466/photo_vae_final.tar"
OUTPUT_PATH = Path("../data/iNaturalistLatentsPhoto")

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

vae_parameters = checkpoint["model_parameters"]
train_parameters = checkpoint["train_parameters"]
train_parameters.device = device


normalize_transform = lambda x: torch.unsqueeze(x.to(torch.float32) / 255, 0).to(device)

vae = VAE(vae_parameters)
vae.load_state_dict(checkpoint["model_state_dict"])
vae = vae.to(train_parameters.device)
vae.eval()

files = (
    list(DATASET_PATH.glob("**/*.jpg"))
    + list(DATASET_PATH.glob("**/*.png"))
    + list(DATASET_PATH.glob("**/*.jpeg"))
)

with torch.no_grad():
    for file in tqdm(files):
        img = torchvision.io.read_image(file)
        img = normalize_transform(img)

        mean, logvar = vae.encode(img)

        output_path = (OUTPUT_PATH / file.relative_to(DATASET_PATH)).with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input": file,
                "mean": mean.cpu().squeeze(0),
                "logvar": logvar.cpu().squeeze(0),
            },
            output_path,
        )
