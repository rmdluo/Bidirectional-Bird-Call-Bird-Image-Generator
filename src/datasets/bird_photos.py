import os
import logging
import random
import pandas as pd
import torchvision
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BirdPhotos(Dataset):
    """Dataset that pairs bird calls and bird photos"""

    def __init__(
        self,
        photo_dir,
        split="train",
        photo_transform=None,
        target_transform=None,
        max_pairs=None,
        **kwargs,
    ):
        """PyTorch dataset for bird images.

        Positional arguments:
        photo_dir -- the directory that holds the photo samples. Assumes the photo files are partitioned by species.

        Keyword arguments:
        random_seed -- random seed to use, used in sampling for one-to-many, many-to-one, and many-to-many. Defaults to None.
        """
        logger.info(
            f"Constructing BirdPhotos dataset from photo directory `{photo_dir}`."
        )
        self.photo_dir = photo_dir
        self.photo_transform, self.target_transform = photo_transform, target_transform

        # Extract species
        photo_species = set(
            [
                name
                for name in os.listdir(os.path.join(photo_dir, split))
                if os.path.isdir(os.path.join(photo_dir, split, name))
            ]
        )
        logger.info(f"Found {len(photo_species)} species.")

        # Generate pairings
        self.pairs = {
            "species": [],
            "photo_filepath": [],
        }
        for species in sorted(photo_species):
            species_photo_files = sorted(
                os.listdir(os.path.join(photo_dir, split, species))
            )
            for photo_file in species_photo_files:
                self.pairs["species"].append(species)
                self.pairs["photo_filepath"].append(
                    os.path.join(photo_dir, split, species, photo_file)
                )
                if (
                    max_pairs is not None
                    and len(self.pairs["photo_filepath"]) >= max_pairs
                ):
                    break
            if max_pairs is not None and len(self.pairs["photo_filepath"]) >= max_pairs:
                break

        self.pairs = pd.DataFrame(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        img = torchvision.io.read_image(pair["photo_filepath"])
        species = pair["species"]

        if self.photo_transform is not None:
            img = self.photo_transform(img)
        if self.target_transform is not None:
            species = self.target_transform(species)

        return img, species
