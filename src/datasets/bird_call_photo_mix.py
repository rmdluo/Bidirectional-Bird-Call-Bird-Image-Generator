import os
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BirdCallPhotoMix(Dataset):
    """Dataset that pairs bird calls and bird photos"""

    def __init__(self, audio_dir, photo_dir, mixing_mode="one-to-one", verbose=False):
        """PyTorch dataset that links audio to photos, matched on species.

        Positional arguments:
        audio_dir -- the directory that holds the audio samples. Assumes the audio files are partitioned by species.
        photo_dir -- the directory that holds the photo samples. Assumes the photo files are partitioned by species.

        Keyword arguments:
        mixing_mode -- whether to use one-to-one, one-to-many, many-to-one, or many-to-many pairings
            - one-to-one: each audio sample gets paired to a single image
            - one-to-many: each audio sample gets paired to multiple images, but each image is only paired to one audio sample
            - many-to-one: each audio sample gets paired to one image, but each image can be paired to multiple audio samples
            - many-to-many: no constraints, both audio and photo samples may be paired with multiple of the other modality
        verbose -- set to true to increase log level, otherwise no logs are outputted
        """
        logger.info(f"Constructing BirdCallPhotoMix dataset from audio directory `{audio_dir}` and photo directory `{photo_dir}`.")
        self.audio_dir, self.photo_dir = audio_dir, photo_dir

        # Extract species
        audio_species = set(os.listdir(audio_dir))
        photo_species = set(os.listdir(photo_dir))
        shared_species = audio_species.intersection(photo_species)
        logger.info(f"Found {len(shared_species)} shared species.")

        # Generate pairings
        self.pairings = []
        for species in sorted(shared_species):
            if mixing_mode == "one-to-one":
                

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
